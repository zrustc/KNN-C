data_path=.../perfect_data

for params in 'cr 2 4' 'mr 2 4' 'SST-2 2 4' 'sst-5 5 4' 'subj 2 4' 'trec 6 4' 'cb 3 5' 'mrpc 2 5' 'qnli 2 5' 'qqp 2 5' 'rte 2 5' 'wic 2 3'
do 
    set -- $params
    task=$1
    labels=$2
    tindex=$3
    echo $params

    mkdir -p $data_path/$task/prompt
    mkdir -p $data_path/$task/prompt_with_demon
    
    for seed in 100 13 42 87 21
    do 
        echo $seed
        echo "prompt mode"
        python3 data_process.py --train $data_path/$task/16-$seed/train.json --valid $data_path/$task/16-$seed/valid.json --test $data_path/$task/16-$seed/test.json --output $data_path/$task/prompt/$task.seed$seed --mode 1 --task $task --tindex $tindex --num-label $labels 

        echo "prompt+demon mode"
        python3 data_process.py --train $data_path/$task/16-$seed/train.json --valid $data_path/$task/16-$seed/valid.json --test $data_path/$task/16-$seed/test.json --output $data_path/$task/prompt_with_demon/$task.seed$seed --mode 2 --task $task --tindex $tindex --num-label $labels --seed $seed --num-demon-train 16 --num-demon-test 16

    done 
done 

