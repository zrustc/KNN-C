import os
import argparse
import math
import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="KNN Search for GLUE dataset")

    parser.add_argument("--input", type=str, help="train data")
    parser.add_argument("--model-dir", type=str, help="the path of pre-train model")
    parser.add_argument("--task", type=str, help="the task name")
    parser.add_argument("--mode", type=int, help="0:basic, 1:prompt, 2:verbalizers, 3:average")

    parser.add_argument("--num-labels", type=int, help="the label number of task")
    parser.add_argument("--sampled-num", type=int, default=1, help="the sampled number of demon")
    parser.add_argument("--ensemble-num", type=int, default=1, help="the sampled number of demon")
    parser.add_argument("--prompt-num", type=int, default=1, help="the total number of templete")
    parser.add_argument("--knn-k", type=int, default=8, help="the number of topk")
    parser.add_argument("--knn-T", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--hidden-size", type=int, default=1024, help="hidden size")
    parser.add_argument("--max-length", type=int, default=512, help="the max length of input")

    return parser.parse_args()

# For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
def get_verbalizers_ids(task, tindex, tokenizer):
    if task in ['mr', 'cr', 'SST-2']:
        word_list = [" terrible", " great"]
    elif task == 'sst-5':
        word_list = [" terrible", " bad", " okay", " good", " great"]
    elif task == 'subj':
        word_list = [" subjective", " objective"]
    elif task == 'trec':
        word_list = [" Description", " Entity", " Expression", " Human", " Location", " Number"]
    elif task == 'rte':
        if tindex == 4:
            word_list = [" true", " false"]
        word_list = [" Yes", " No"]
    elif task == 'cb':
        if tindex == 4:
            word_list = [' true', ' false', ' neither']
        word_list = [" Yes", " No", " Maybe"]
    elif task == 'wic':
        if tindex == 2:
            word_list = ["2", "b"]
        word_list = [" No", " Yes"]
    elif task == 'qnli':
        if tindex in [0, 2, 4]:
            word_list = [" Yes", " No"]
        word_list = [" true", " false"]
    elif task in ['qqp', 'mrpc']:
        if tindex in [0, 2, 4]:
            word_list = [" No", " Yes"]
        word_list = [" false", " true"]
    else:
        assert "error task name....."

    return [tokenizer(word, add_special_tokens=False)['input_ids'][0] for word in word_list]

def load_dataset(file_name, sampleNum=1):
    data_inputs = []
    data_labels = []
    with open(file_name, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            items = line.strip().split('\t')
            for index in range(sampleNum):
                data_inputs.append(items[index])
                data_labels.append(int(items[-1]))

    return data_inputs, data_labels

def load_datastore(save_path, train_inputs, train_labels, batch_size,
            hidden_size, get_vec_fun, label_word_ids):
    if os.path.exists(save_path + '.keys.npy'):
        datastore_keys = np.load(save_path + '.keys.npy')
        datastore_vals = np.load(save_path + '.labels.npy')
        datastore_probs = np.load(save_path + '.probs.npy')
    else:
        datastore_keys, datastore_vals, datastore_probs = build_datastore(train_inputs, \
            train_labels, batch_size, hidden_size, get_vec_fun, label_word_ids)
        np.save(save_path + '.keys', datastore_keys) 
        np.save(save_path + '.labels', datastore_vals) 
        np.save(save_path + '.probs', datastore_probs) 
    
    datastore_keys = torch.from_numpy(datastore_keys).to(DEVICE)
    datastore_vals = torch.from_numpy(datastore_vals).to(DEVICE)
    datastore_probs = torch.from_numpy(datastore_probs).to(DEVICE)

    return datastore_keys, datastore_vals, datastore_probs

def get_results_with_prompt(sents, model, tokenizer, max_length, label_word_ids):
    with torch.no_grad():
        inputs = tokenizer(sents, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
        # inputs['token_type_ids'] = inputs['token_type_ids'].to(DEVICE)
        inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)
        results = model(**inputs, output_hidden_states=True)
        hidden_states = results.hidden_states[-1]
        logits = results.logits
        label_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)   # label_idx = (batch_index_tensor, sequence_index_tensor)
        output_hidden_state = torch.zeros([len(sents), hidden_states.size()[-1]], dtype=torch.float32, device=hidden_states.device)
        output_logits = torch.zeros([len(sents), len(label_word_ids)], dtype=torch.float32, device=hidden_states.device)
        for index in range(len(label_idx[0])):
            index_0 = label_idx[0][index]
            index_1 = label_idx[1][index]
            output_hidden_state[index,:] = hidden_states[index_0, index_1, :]
            for word_index, word_id in enumerate(label_word_ids):
                output_logits[index, word_index] = logits[index_0, index_1, word_id]
        return output_hidden_state, torch.softmax(output_logits, dim=-1)

def build_datastore(data_inputs, data_labels, batch_size, hidden_size, func_get_vec, label_word_ids):
    total_len = len(data_inputs)
    datastore_keys = np.zeros([total_len, hidden_size], dtype=np.float32)
    datastore_vals = np.zeros([total_len], dtype=np.int)
    datastore_probs = np.zeros([total_len, len(label_word_ids)], dtype=np.float32)

    start = 0
    for start in range(0, total_len, batch_size):
        end = min(total_len, start + batch_size)
        vecs, probs = func_get_vec(data_inputs[start:end], label_word_ids)
        datastore_keys[start:end] = vecs.cpu().numpy()
        datastore_vals[start:end] = data_labels[start:end]
        datastore_probs[start:end] = probs.cpu().numpy()

    return datastore_keys, datastore_vals, datastore_probs

def calculate_topk_prob(queries, queries_probs, keys, values, knn_T, 
            knn_k, knn_lambda, batch_size, num_labels):
    # queries [B, H]
    # keys [L, H]
    dists = ((keys.unsqueeze(0) - queries.unsqueeze(1)) ** 2).sum(-1) # [B, L]
    scaled_dists = -1.0 / knn_T * dists
    top_dists, top_indices = torch.topk(scaled_dists, knn_k) # [B, K]
    new_vals = values.unsqueeze(0).repeat(batch_size, 1)
    top_values = torch.gather(new_vals, 1, top_indices).unsqueeze(-1)
    knn_weight = torch.softmax(top_dists, dim=-1).unsqueeze(-1)  # [B, K, 1]

    knn_tgt_prob = torch.zeros([batch_size, knn_k, num_labels], dtype=torch.float32).to(DEVICE)
    knn_tgt_prob.scatter_(2, top_values, knn_weight)
    knn_prob = knn_tgt_prob.sum(dim=-2)  # [B, V]
    final_prob = queries_probs * (1.0 - knn_lambda) + knn_prob * knn_lambda

    return final_prob

def evaluate_dataset(train_ds_inputs, test_ds_inputs, test_labels,
        knn_T, knn_k, knn_lambda, batch_size, num_labels, ensemble_num=1):

    train_ds_keys, train_ds_vals, train_ds_probs = train_ds_inputs
    test_ds_keys, test_ds_vals, test_ds_probs = test_ds_inputs
    total_len = len(test_labels)
    correct = 0
    for start in range(0, total_len, batch_size * ensemble_num):
        with torch.no_grad():
            end = min(total_len, start + batch_size * ensemble_num)
            cur_batch = (end - start) // ensemble_num
            test_ids = torch.Tensor([test_labels[index] for index in range(start, end, ensemble_num)]).to(DEVICE)
            test_vecs = test_ds_keys[start:end]
            test_probs = test_ds_probs[start:end]
            final_probs = calculate_topk_prob(test_vecs, test_probs, \
                train_ds_keys, train_ds_vals, knn_T, knn_k, knn_lambda, end - start, num_labels)
            if ensemble_num > 1:
                final_probs = final_probs.reshape([cur_batch, -1, num_labels])
                total_probs = torch.sum(final_probs, dim=-2) / ensemble_num
            else:
                total_probs = final_probs
            predict_ids = total_probs.argmax(axis=-1)
            correct += torch.eq(predict_ids, test_ids).int().sum().detach().cpu().numpy()      

    return 1.0 * correct * ensemble_num / total_len

def main(args):
    # load pre-trained model
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_dir)
    model = RobertaForMaskedLM.from_pretrained(args.model_dir).to(DEVICE)
    model.eval()

    # load hyper-parameters
    sampled_num = args.sampled_num
    prompt_num = args.prompt_num
    knn_k = args.knn_k
    knn_T = args.knn_T
    batch_size = args.batch_size
    num_labels = args.num_labels
    ensemble_num = args.ensemble_num
    hidden_size = args.hidden_size

    all_seeds = [13, 21, 42, 87, 100]

    # build map function
    def get_vec_fun(sents, label_word_ids):
        return get_results_with_prompt(sents, model, tokenizer, args.max_length, label_word_ids)

    valid_std = np.zeros(prompt_num * len(all_seeds), dtype=np.float32)
    valid_std_0 = np.zeros(prompt_num * len(all_seeds), dtype=np.float32)
    valid_std_1 = np.zeros(prompt_num * len(all_seeds), dtype=np.float32)

    test_std = np.zeros(prompt_num * len(all_seeds), dtype=np.float32)
    test_std_0 = np.zeros(prompt_num * len(all_seeds), dtype=np.float32)
    test_std_1 = np.zeros(prompt_num * len(all_seeds), dtype=np.float32)
    
    task_index = 0
    for seed in all_seeds:
        for tindex in range(prompt_num):
            # load datasets
            train_inputs, train_labels = load_dataset('{0}.seed{1}.tindex{2}.train'.format(args.input, seed, tindex), sampled_num)
            valid_inputs, valid_labels = load_dataset('{0}.seed{1}.tindex{2}.valid'.format(args.input, seed, tindex), ensemble_num)
            test_inputs, test_labels = load_dataset('{0}.seed{1}.tindex{2}.test'.format(args.input, seed, tindex), ensemble_num)
            label_word_ids = get_verbalizers_ids(args.task, tindex, tokenizer)

            # build datastore
            save_path = '{0}.seed{1}.tindex{2}.train.s{3}'.format(args.input, seed, tindex, sampled_num)
            train_ds_inputs = load_datastore(save_path, train_inputs, train_labels, \
                    batch_size, hidden_size, get_vec_fun, label_word_ids)

            save_path = '{0}.seed{1}.tindex{2}.valid.s{3}'.format(args.input, seed, tindex, ensemble_num)
            valid_ds_inputs = load_datastore(save_path, valid_inputs, valid_labels, \
                    batch_size, hidden_size, get_vec_fun, label_word_ids)

            save_path = '{0}.seed{1}.tindex{2}.test.s{3}'.format(args.input, seed, tindex, ensemble_num)
            test_ds_inputs = load_datastore(save_path, test_inputs, test_labels,  \
                    batch_size, hidden_size, get_vec_fun, label_word_ids)
            
            # evaluate 
            valid_acc_0 = evaluate_dataset(train_ds_inputs, valid_ds_inputs, valid_labels, \
                            knn_T, knn_k, 0, batch_size, num_labels, ensemble_num)
            test_acc_0 = evaluate_dataset(train_ds_inputs, test_ds_inputs, test_labels, \
                            knn_T, knn_k, 0, batch_size, num_labels, ensemble_num)
            valid_acc_1 = evaluate_dataset(train_ds_inputs, valid_ds_inputs, valid_labels, \
                            knn_T, knn_k, 1, batch_size, num_labels, ensemble_num)
            test_acc_1 = evaluate_dataset(train_ds_inputs, test_ds_inputs, test_labels, \
                            knn_T, knn_k, 1, batch_size, num_labels, ensemble_num)

            best_knn_lambda = 0
            valid_acc = 0
            for knn_lambda in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                valid_acc_tmp = evaluate_dataset(train_ds_inputs, valid_ds_inputs, valid_labels, \
                            knn_T, knn_k, knn_lambda, batch_size, num_labels, ensemble_num)
                if valid_acc_tmp > valid_acc:
                    best_knn_lambda = knn_lambda
                    valid_acc = valid_acc_tmp
            test_acc = evaluate_dataset(train_ds_inputs, test_ds_inputs, test_labels, \
                            knn_T, knn_k, best_knn_lambda, batch_size, num_labels, ensemble_num)

            valid_std_0[task_index] = valid_acc_0
            valid_std_1[task_index] = valid_acc_1
            valid_std[task_index] = valid_acc

            test_std_0[task_index] = test_acc_0
            test_std_1[task_index] = test_acc_1
            test_std[task_index] = test_acc
            task_index += 1

            print('topk={0},tempreture={1},seed={2},tindex={3} ICL --> valid acc: {4}, test acc: {5}'.format(str(knn_k), str(knn_T), str(seed), str(tindex), '{:.2f}'.format(100 * valid_acc_0), '{:.2f}'.format(100 * test_acc_0)))
            print('topk={0},tempreture={1},seed={2},tindex={3} KNN-C w/o ANS,FR,P_LM--> valid acc: {4}, test acc: {5}'.format(str(knn_k), str(knn_T), str(seed), str(tindex), '{:.2f}'.format(100 * valid_acc_1), '{:.2f}'.format(100 * test_acc_1)))
            print('topk={0},tempreture={1},seed={2},tindex={3} KNN-C w/o ANS,FR --> valid acc: {4}, test acc: {5}'.format(str(knn_k), str(knn_T), str(seed), str(tindex), '{:.2f}'.format(100 * valid_acc), '{:.2f}'.format(100 * test_acc)))

    print('topk={0}, tempreture={1} ICL --> avg/min valid acc: {2}/{3}, avg/min/std test acc: {4}/{5}/{6}'.format(str(knn_k), str(knn_T), '{:.2f}'.format( 100 * np.mean(valid_std_0)), '{:.2f}'.format(100 * np.amin(valid_std_0)), '{:.2f}'.format(100 * np.mean(test_std_0)), '{:.2f}'.format(100 * np.amin(test_std_0)), '{:.2f}'.format(100 * np.std(test_std_0))))
    print('topk={0}, tempreture={1} KNN-C w/o ANS,FR,P_LM --> avg/min valid acc: {2}/{3}, avg/min/std test acc: {4}/{5}/{6}'.format(str(knn_k), str(knn_T), '{:.2f}'.format( 100 * np.mean(valid_std_1)), '{:.2f}'.format(100 * np.amin(valid_std_1)), '{:.2f}'.format(100 * np.mean(test_std_1)), '{:.2f}'.format(100 * np.amin(test_std_1)), '{:.2f}'.format(100 * np.std(test_std_1))))
    print('topk={0}, tempreture={1} KNN-C w/o ANS,FR --> avg/min valid acc: {2}/{3}, avg/min/std test acc: {4}/{5}/{6}'.format(str(knn_k), str(knn_T), '{:.2f}'.format( 100 * np.mean(valid_std)), '{:.2f}'.format(100 * np.amin(valid_std)), '{:.2f}'.format(100 * np.mean(test_std)), '{:.2f}'.format(100 * np.amin(test_std)), '{:.2f}'.format(100 * np.std(test_std))))


if __name__ == '__main__':
    main(parse_args())
