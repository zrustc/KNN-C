import os
import argparse
import math
import torch
import torch.nn as nn
import numpy as np
import copy
import random
from torch_scatter import scatter
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
import torch.optim as optim
from torch.autograd import Variable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="KNN Search for GLUE dataset")

    parser.add_argument("--input", type=str, help="train data")
    parser.add_argument("--model-dir", type=str, help="the path of pre-train model")
    parser.add_argument("--task", type=str, help="the task name")

    parser.add_argument("--num-labels", type=int, help="the label number of task")
    parser.add_argument("--sampled-num", type=int, default=1, help="the sampled number of demon")
    parser.add_argument("--ensemble-num", type=int, default=1, help="the ensemble number of demon")
    parser.add_argument("--prompt-num", type=int, default=1, help="the total number of templete")
    parser.add_argument("--knn-k", type=int, default=8, help="the number of topk")
    parser.add_argument("--knn-T", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--max-length", type=int, default=512, help="the max length of input")
    parser.add_argument("--hidden-size", type=int, default=1024, help="hidden size")
    parser.add_argument("--train-epoch", type=int, default=30, help="batch size")
    parser.add_argument("--map-size", type=int, default=32, help="mapping size for feature compression")

    return parser.parse_args()

class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        batch_idx = torch.arange(labels.shape[0], device=logits.device)
        loss = log_probs[batch_idx, labels]
        n = logits.shape[-1] - 1.0
        p = 1.0 - self.smoothing
        q = self.smoothing / n
        sum_probs = torch.sum(log_probs, dim=-1)
        loss = p * loss + q * (sum_probs - loss)
        return -loss.sum()

'''
feature mapping + compression
'''
class MetaKNetwork(nn.Module):
    def __init__(self, input_size, map_size, output_size):
        super().__init__()
        self.f1_func = nn.Linear(input_size, 32)
        self.f2_func = nn.Linear(32, output_size)     # select_topk
        self.dropout = nn.Dropout(p=0.3)
        self.loss = SmoothedCrossEntropyLoss(smoothing=0.1)
        self.reset_parameters()

    def forward(self, x):
        hidden = torch.relu(self.f1_func(self.dropout(x)))
        select_topk_prob = torch.softmax(self.f2_func(self.dropout(hidden)), dim=-1)
        return select_topk_prob

    def reset_parameters(self):
        # f1_func
        nn.init.kaiming_uniform_(self.f1_func.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.f1_func.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.f1_func.bias, -bound, bound)
        # f2_func
        nn.init.kaiming_uniform_(self.f2_func.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.f2_func.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.f2_func.bias, -bound, bound)

class CompactLayer(nn.Module):
    def __init__(self, hidden_size, map_size):
        super().__init__()
        self.f1_func = nn.Linear(hidden_size, map_size)
        self.dropout = nn.Dropout(p=0.3)
        self.loss = SmoothedCrossEntropyLoss(smoothing=0.1)
        self.reset_parameters()

    def forward(self, x):
        hidden = torch.relu(self.f1_func(self.dropout(x)))
        return hidden

    def reset_parameters(self):
        # f1_func
        nn.init.kaiming_uniform_(self.f1_func.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.f1_func.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.f1_func.bias, -bound, bound)

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
            hidden_size, get_vec_fun, label_word_ids, reuse=True):
    if os.path.exists(save_path + '.keys.npy') and reuse:
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

    for start in range(0, total_len, batch_size):
        end = min(total_len, start + batch_size)
        vecs, probs = func_get_vec(data_inputs[start:end], label_word_ids)
        datastore_keys[start:end] = vecs.cpu().numpy()
        datastore_vals[start:end] = data_labels[start:end]
        datastore_probs[start:end] = probs.cpu().numpy()

    return datastore_keys, datastore_vals, datastore_probs

def calculate_topk_prob(queries, keys, values, knn_T, knn_k, batch_size, 
            num_labels, isTrain=False, isRemoveTop1=False):
    # queries [B, H]
    # keys [L, H]
    dists = ((keys.unsqueeze(0) - queries.unsqueeze(1)) ** 2).sum(-1) # [B, L]
    scaled_dists = -1.0 / knn_T * dists
    top_dists, top_indices = torch.topk(scaled_dists, (knn_k + 1) if isRemoveTop1 else knn_k) # [B, K + 1]
    new_vals = values.unsqueeze(0).repeat(batch_size, 1)
    top_values = torch.gather(new_vals, 1, top_indices[:, 1:] if isRemoveTop1 else top_indices).unsqueeze(-1) 
    knn_weight = torch.softmax(top_dists[:, 1:] if isRemoveTop1 else top_dists, dim=-1).unsqueeze(-1)  # [B, K, 1]
    
    # init knn-prob
    knn_tgt_prob = torch.zeros([batch_size, knn_k, num_labels], dtype=torch.float32, device=keys.device)
    if isTrain:
        knn_tgt_prob = Variable(knn_tgt_prob).clone()
    knn_tgt_prob.scatter_(2, top_values, knn_weight)
    prob = knn_tgt_prob.sum(dim=-2)  # [B, V]

    return prob

def calculate_adaptive_topk_prob(queries, queries_probs, keys, values, knn_T,
            knn_k_list, batch_size, num_labels, isRemoveTop1=False):
    # queries [B, H]
    # keys [L, H]
    dists = ((keys.unsqueeze(0) - queries.unsqueeze(1)) ** 2).sum(-1) # [B, L]
    scaled_dists = -1.0 / knn_T * dists
    top_dists, top_indices = torch.topk(scaled_dists, 
        knn_k_list[-1] + 1) # [B, K_max]
    new_vals = values.unsqueeze(0).repeat(batch_size, 1)
    top_values = torch.gather(new_vals, 1, top_indices) # [B, K_max]
    if isRemoveTop1:
        top_dists = top_dists[:, 1:] # [B, K_max]

    knn_prob_list = []
    for topk in knn_k_list:
        if topk == 0:
            knn_prob_list.append(queries_probs)
        else:
            knn_weight_k = torch.softmax(top_dists[:, :topk], dim=-1).unsqueeze(-1)  # [B, K, 1]
            top_values_k = top_values[:, :topk].unsqueeze(-1)
            knn_tgt_prob = torch.zeros([batch_size, topk, num_labels], dtype=torch.float32, device=queries.device)
            knn_tgt_prob.scatter_(2, top_values_k, knn_weight_k)
            knn_prob = knn_tgt_prob.sum(dim=-2)  # [B, V]
            knn_prob_list.append(knn_prob)
    
    knn_prob = torch.stack(knn_prob_list, dim=1) # [B, k_num, V]

    # count label number for top-k
    top_values_cpu = top_values.cpu()
    top_values_count = torch.zeros_like(top_values_cpu) 
    for index in range(batch_size):
        value_set = set()
        for k_index in range(knn_k_list[-1]):
            value_set.add(top_values[index, k_index])
            top_values_count[index, k_index] = len(value_set)
    top_values_count = top_values_count.to(DEVICE) # [B, K]

    selected_topk_dists = torch.zeros([batch_size, len(knn_k_list)], dtype=torch.float32, device=queries.device)
    selected_topk_count = torch.zeros([batch_size, len(knn_k_list)], dtype=torch.float32, device=queries.device)
    for index, topk in enumerate(knn_k_list):
        selected_topk_dists[:,index] = top_dists[:,topk]
        selected_topk_count[:,index] = top_values_count[:,topk]
    
    return knn_prob, selected_topk_dists, selected_topk_count

def evaluate_dataset(train_ds_inputs, test_ds_inputs, test_labels, metakLayer, 
        knn_T, knn_k, batch_size, num_labels, ensemble_num, isAdaptive=True):

    topk_list = [0, 4, 8, 12, 16]
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
            if isAdaptive:
                test_probs = test_ds_probs[start:end]
                knn_probs, top_dists, top_values_count = calculate_adaptive_topk_prob(test_vecs, test_probs, \
                    train_ds_keys, train_ds_vals, knn_T, topk_list, end - start, num_labels)
                input_feature = torch.hstack((top_dists, top_values_count))
                adaptive_weights = metakLayer(input_feature).unsqueeze(-1)
                knn_probs = (knn_probs * adaptive_weights).sum(dim=1)
            else:
                knn_probs = calculate_topk_prob(test_vecs, train_ds_keys, train_ds_vals, \
                        knn_T, knn_k, end - start, num_labels)
            if ensemble_num > 1:
                knn_probs = knn_probs.reshape([cur_batch, -1, num_labels])
                total_probs = torch.mean(knn_probs, dim=-2)
            else:
                total_probs = knn_probs
            predict_ids = total_probs.argmax(axis=-1)
            correct += torch.eq(predict_ids, test_ids).int().sum().cpu().numpy()      

    return 1.0 * correct * ensemble_num / total_len

def model_training(train_ds_inputs, train_labels, valid_ds_inputs, valid_labels,
        hidden_size, map_size, batch_size, total_epoch, knn_T, knn_k, num_labels,
        sampled_num, ensemble_num=1, train_mode=0):

    topk_list = [0, 4, 8, 12, 16]
    train_ds_keys, train_ds_vals, train_ds_probs = train_ds_inputs
    if train_mode == 0:
        trainLayer = CompactLayer(hidden_size, map_size).to(DEVICE)
    else:
        trainLayer = MetaKNetwork(len(topk_list) * 2, map_size, len(topk_list)).to(DEVICE)

    total_len = len(train_labels)
    # split training set
    train_number = total_len // sampled_num
    total_len = total_len // 2 # haif for buiding datastore and rest for training

    searched_ds_keys = torch.zeros([total_len, hidden_size], dtype=torch.float32, device=train_ds_keys.device)
    searched_ds_vals = torch.zeros([total_len], dtype=torch.int64, device=train_ds_keys.device)
    query_ds_keys = torch.zeros([total_len, hidden_size], dtype=torch.float32, device=train_ds_keys.device)
    query_ds_probs = torch.zeros([total_len, num_labels], dtype=torch.float32, device=train_ds_keys.device)
    query_ds_vals = torch.zeros([total_len], dtype=torch.int64, device=train_ds_keys.device)

    class_tag = [1 - train_mode] * num_labels
    searched_index = 0
    query_index = 0
    for index in range(train_number):
        cur_index = index * sampled_num
        class_id = train_labels[cur_index]
        if class_tag[class_id] == 0:
            searched_ds_keys[searched_index:searched_index+sampled_num] = train_ds_keys[cur_index:cur_index+sampled_num]
            searched_ds_vals[searched_index:searched_index+sampled_num] = train_ds_vals[cur_index:cur_index+sampled_num]
            searched_index += sampled_num
            class_tag[class_id] = 1
        else:
            query_ds_keys[query_index:query_index+sampled_num] = train_ds_keys[cur_index:cur_index+sampled_num]
            query_ds_probs[query_index:query_index+sampled_num] = train_ds_probs[cur_index:cur_index+sampled_num]
            query_ds_vals[query_index:query_index+sampled_num] = train_ds_vals[cur_index:cur_index+sampled_num]
            query_index += sampled_num
            class_tag[class_id] = 0

    best_Layer = None
    best_valid_acc = 0
    optimizer = optim.Adam(trainLayer.parameters())

    for epoch in range(total_epoch):
        # shuffle data
        running_loss = 0.0
        trainLayer.train()
        training_order = torch.randperm(total_len)
        all_input_keys = query_ds_keys[training_order]
        all_input_probs = query_ds_probs[training_order]
        all_input_vals = query_ds_vals[training_order]
        
        for start in range(0, total_len, 64):
            end = min(total_len, start + 64)
            # build train data
            input_vecs = all_input_keys[start:end]
            input_probs = all_input_probs[start:end]
            input_vals = all_input_vals[start:end]
            # zero the parameter gradients
            optimizer.zero_grad()
            if train_mode == 0:
                cur_topk = 8
                input_vecs = trainLayer(input_vecs)
                all_keys = trainLayer(searched_ds_keys)
                knn_probs = calculate_topk_prob(input_vecs, all_keys, searched_ds_vals, knn_T, cur_topk, \
                        end - start, num_labels, True)
                loss = trainLayer.loss(torch.log(knn_probs + 1e-20), input_vals)
            else:
                knn_probs, top_dists, top_values_count = calculate_adaptive_topk_prob(input_vecs, input_probs, \
                        searched_ds_keys, searched_ds_vals, knn_T, topk_list, end - start, num_labels)
                input_feature = torch.hstack((top_dists, top_values_count))
                adaptive_weights = trainLayer(input_feature).unsqueeze(-1)
                knn_probs = (knn_probs * adaptive_weights).sum(dim=1)
                loss = trainLayer.loss(torch.log(knn_probs + 1e-20), input_vals)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainLayer.parameters(), 10)
            optimizer.step()
            running_loss += loss.item()

        # valid 
        trainLayer.eval()
        if train_mode == 0:
            with torch.no_grad():
                train_ds_inputs_new = (trainLayer(train_ds_inputs[0]), train_ds_inputs[1], train_ds_inputs[2])
                valid_ds_inputs_new = (trainLayer(valid_ds_inputs[0]), valid_ds_inputs[1], valid_ds_inputs[2])
        else:
            train_ds_inputs_new = train_ds_inputs
            valid_ds_inputs_new = valid_ds_inputs
        valid_acc = evaluate_dataset(train_ds_inputs_new, valid_ds_inputs_new, valid_labels, trainLayer, \
                        knn_T, knn_k, batch_size, num_labels, 1, False if train_mode == 0 else True)

        if valid_acc > best_valid_acc:
            best_Layer = copy.deepcopy(trainLayer)
            best_valid_acc = valid_acc

    return best_Layer, best_valid_acc

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

    map_size = args.map_size
    train_epoch = args.train_epoch

    all_seeds = [13, 21, 42, 87, 100]

    # build map function
    def get_vec_fun(sents, label_word_ids=None):
        return get_results_with_prompt(sents, model, tokenizer, args.max_length, label_word_ids)

    test_ensemble_num_list = [ensemble_num if ensemble_num > 1 else 1]
    valid_std_fm = np.zeros(prompt_num * len(all_seeds) * 4, dtype=np.float32)
    valid_std_all = np.zeros(prompt_num * len(all_seeds) * 4, dtype=np.float32)
    test_std_fm_list = [0] * len(test_ensemble_num_list)
    test_std_all_list = [0] * len(test_ensemble_num_list)
    for index in range(len(test_ensemble_num_list)):
        test_std_fm_list[index] = np.zeros(prompt_num * len(all_seeds) * 4, dtype=np.float32)
        test_std_all_list[index] = np.zeros(prompt_num * len(all_seeds) * 4, dtype=np.float32)

    task_index = 0
    for seed in all_seeds:
        for tindex in range(prompt_num):
            # load datasets
            train_inputs, train_labels = load_dataset('{0}.seed{1}.tindex{2}.train'.format(args.input, seed, tindex), sampled_num)
            valid_inputs, valid_labels = load_dataset('{0}.seed{1}.tindex{2}.valid'.format(args.input, seed, tindex), ensemble_num)
            label_word_ids = get_verbalizers_ids(args.task, tindex, tokenizer)

            for train_seed in [1, 10, 100, 1000]:
                random.seed(train_seed)
                np.random.seed(train_seed)
                torch.manual_seed(train_seed)

                save_path = '{0}.seed{1}.tindex{2}.train.s{3}'.format(args.input, seed, tindex, sampled_num)
                train_ds_inputs = load_datastore(save_path, train_inputs, train_labels, \
                    batch_size, hidden_size, get_vec_fun, label_word_ids)

                save_path = '{0}.seed{1}.tindex{2}.valid.s{3}'.format(args.input, seed, tindex, ensemble_num)
                valid_ds_inputs = load_datastore(save_path, valid_inputs, valid_labels, \
                        batch_size, hidden_size, get_vec_fun, label_word_ids)

                # model training
                compactLayer, valid_acc_fm = model_training(train_ds_inputs, train_labels, valid_ds_inputs, valid_labels,
                        hidden_size, map_size, batch_size, train_epoch, knn_T, knn_k, num_labels,
                        sampled_num, ensemble_num, 0)
                compactLayer.eval()
                valid_std_fm[task_index] = valid_acc_fm

                # build new feature function
                with torch.no_grad():
                    train_ds_inputs = (compactLayer(train_ds_inputs[0]), train_ds_inputs[1], train_ds_inputs[2])
                    valid_ds_inputs = (compactLayer(valid_ds_inputs[0]), valid_ds_inputs[1], valid_ds_inputs[2])

                for test_index, test_ensemble_num in enumerate(test_ensemble_num_list):
                    test_inputs, test_labels = load_dataset('{0}.seed{1}.tindex{2}.test'.format(args.input, seed, tindex), test_ensemble_num)
                    save_path = '{0}.seed{1}.tindex{2}.test.s{3}'.format(args.input, seed, tindex, test_ensemble_num)
                    test_ds_inputs = load_datastore(save_path, test_inputs, test_labels, batch_size, \
                                        hidden_size, get_vec_fun, label_word_ids, False if args.task == 'qqp' and train_seed == 1 else True) 
                    with torch.no_grad():
                        test_ds_inputs = (compactLayer(test_ds_inputs[0]), test_ds_inputs[1], test_ds_inputs[2])
                    # evaluate 
                    test_acc_fm = evaluate_dataset(train_ds_inputs, test_ds_inputs, test_labels, None, \
                            knn_T, knn_k, batch_size, num_labels, test_ensemble_num, False)
                    test_std_fm_list[test_index][task_index] = test_acc_fm

                    print('topk={0},tempreture={1},seed={2},tindex={3},train_seed={4},demon_num={5} KNN-C (FR) --> valid acc: {6}, test acc: {7}'.format(str(knn_k), str(knn_T), str(seed), str(tindex), str(train_seed), str(test_ensemble_num), '{:.2f}'.format(100 * valid_acc_fm), '{:.2f}'.format(100 * test_acc_fm)))
                
                metakLayer, valid_acc = model_training(train_ds_inputs, train_labels, valid_ds_inputs, valid_labels,
                        map_size, map_size, batch_size, train_epoch, knn_T, knn_k, num_labels,
                        sampled_num, ensemble_num, 1)
                metakLayer.eval()
                valid_std_all[task_index] = valid_acc

                for test_index, test_ensemble_num in enumerate(test_ensemble_num_list):
                    test_inputs, test_labels = load_dataset('{0}.seed{1}.tindex{2}.test'.format(args.input, seed, tindex), test_ensemble_num)
                    save_path = '{0}.seed{1}.tindex{2}.test.s{3}'.format(args.input, seed, tindex, test_ensemble_num)
                    test_ds_inputs = load_datastore(save_path, test_inputs, test_labels, batch_size, \
                                        hidden_size, get_vec_fun, label_word_ids)
                    with torch.no_grad():
                        test_ds_inputs = (compactLayer(test_ds_inputs[0]), test_ds_inputs[1], test_ds_inputs[2])
                    # evaluate 
                    test_acc = evaluate_dataset(train_ds_inputs, test_ds_inputs, test_labels, metakLayer, \
                            knn_T, knn_k, batch_size, num_labels, test_ensemble_num, True)
                    test_std_all_list[test_index][task_index] = test_acc
              
                    print('topk={0},tempreture={1},seed={2},tindex={3},train_seed={4},demon_num={5} KNN-C (ANS+FR) --> valid acc: {6}, test acc: {7}'.format(str(knn_k), str(knn_T), str(seed), str(tindex), str(train_seed), str(test_ensemble_num), '{:.2f}'.format(100 * valid_acc), '{:.2f}'.format(100 * test_acc)))

                task_index += 1

    print("-----------------total--------------------")
    for test_index, test_ensemble_num in enumerate(test_ensemble_num_list):
        print('final || topk={0},tempreture={1},demon_num={2} KNN-C (FR) --> avg/min valid acc: {3}/{4}, avg/min/std test acc: {5}/{6}/{7}'.format(str(knn_k), str(knn_T), str(test_ensemble_num), '{:.2f}'.format(100 * np.mean(valid_std_fm)), '{:.2f}'.format(100 * np.amin(valid_std_fm)), '{:.2f}'.format(100 * np.mean(test_std_fm_list[test_index])), '{:.2f}'.format(100 * np.amin(test_std_fm_list[test_index])), '{:.2f}'.format(100 * np.std(test_std_fm_list[test_index]))))

        print('final || topk={0},tempreture={1},demon_num={2} KNN-C (ANS+FR) --> avg/min valid acc: {3}/{4}, avg/min/std test acc: {5}/{6}/{7}'.format(str(knn_k), str(knn_T), str(test_ensemble_num), '{:.2f}'.format(100 * np.mean(valid_std_all)), '{:.2f}'.format(100 * np.amin(valid_std_all)), '{:.2f}'.format(100 * np.mean(test_std_all_list[test_index])), '{:.2f}'.format(100 * np.amin(test_std_all_list[test_index])), '{:.2f}'.format(100 * np.std(test_std_all_list[test_index]))))
        
if __name__ == '__main__':
    main(parse_args())
