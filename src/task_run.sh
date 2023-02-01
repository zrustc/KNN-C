data_path=...
code_path=...
log_path=...

device_id=$1
task_name=$2
labels=$3
prompts=$4
log_name=$5

log_file=$log_path/$task_name/$log_name.log
mkdir -p $log_path/$task_name
rm $log_file

echo $task_name

model_path=.../roberta-large

echo "-------------------->"                        >>  $log_file
echo "Zero-shot (PV-Zero)"                    >>  $log_file
echo "-------------------->"                        >>  $log_file
input_data=$data_path/$task_name/prompt/$task_name
CUDA_VISIBLE_DEVICES=$device_id python3 $code_path/icl_roberta.py --input $input_data --model-dir $model_path --task $task_name --num-labels $labels --prompt-num $prompts --sampled-num 1 --ensemble-num 1 --batch-size 64 >> $log_file

echo "-------------------->"                        >>  $log_file
echo "In-context Learning (ICL)"                    >>  $log_file
echo "-------------------->"                        >>  $log_file
input_data=$data_path/$task_name/prompt_with_demon/$task_name
CUDA_VISIBLE_DEVICES=$device_id python3 $code_path/icl_roberta.py --input $input_data --model-dir $model_path --task $task_name --num-labels $labels --prompt-num $prompts --sampled-num 17 --ensemble-num 17 --batch-size 64 >> $log_file

echo "-------------------->"                        >>  $log_file
echo "Nearest Neighbor Calibration (KNN-C)"         >>  $log_file
echo "-------------------->"                        >>  $log_file
input_data=$data_path/$task_name/prompt_with_demon/$task_name
CUDA_VISIBLE_DEVICES=$device_id python3 $code_path/knn_c_roberta.py --input $input_data --model-dir $model_path --task $task_name --num-labels $labels --prompt-num $prompts --sampled-num 17 --ensemble-num 17 --batch-size 64 >> $log_file
