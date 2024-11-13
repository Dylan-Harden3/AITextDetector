#!/bin/bash
models=("google/gemma-2-9b" "meta-llama/Llama-3.1-8B" "tiiuae/falcon-7b")
cache_dir="/scratch/user/dylanharden3/AITextDetector/dataset"

for model in "${models[@]}"; do
    model_repo="$model"
    model_name=$(basename "$model")
    for dataset_model in "${models[@]}"; do
        dataset_name=$(basename "$dataset_model")
        if [ "$model_name" == "$dataset_name" ]; then
            continue
        fi

        dataset_file="${dataset_name}_xsum.json"
        output_file="${model_name}_${dataset_name}_baselines.json"

        echo "Running baslines for model: $model_repo with dataset: $dataset_file"
        python baselines.py --dataset_file "$dataset_file" --model "$model_repo" --output_file "$output_file" --cache "$cache_dir"
    done
done
