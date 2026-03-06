#!/bin/bash

method="PF-GIM-IG"
ablation="patching"
level="edge"
filter_quantile=${1:-0.03}
filter_mode=${2:-proximity}
ig_steps=${3:-5}
eval_split=${4:-validation}

VALID_COMBOS=("interpbench_ioi" "gpt2_ioi" "qwen2.5_ioi" "gemma2_ioi" "llama3_ioi" \
                "qwen2.5_mcqa" "gemma2_mcqa" "llama3_mcqa" \
                "llama3_arithmetic_addition" "llama3_arithmetic_subtraction" \
                "gemma2_arc_easy" "llama3_arc_easy" "llama3_arc_challenge")

# Attribution pass
for model in {interpbench,gpt2,qwen2.5,gemma2,llama3}; do
    for task in {ioi,mcqa,arithmetic_addition,arithmetic_subtraction,arc_easy,arc_challenge}; do

        combo="${model}_${task}"
        valid=false
        for valid_combo in "${VALID_COMBOS[@]}"; do
            if [[ "$combo" == "$valid_combo" ]]; then
                valid=true
                break
            fi
        done
        if [[ "$valid" == "false" ]]; then
            continue
        fi

        echo "=== Attribution: $model on $task ==="

        if [ "$model" = "llama3" ]; then
            batch_size=1
        elif [ "$task" = "arc_easy" ] || [ "$task" = "arc_challenge" ]; then
            batch_size=1
        elif [ "$model" = "gpt2" ] || [ "$model" = "interpbench" ]; then
            batch_size=20
        else
            batch_size=10
        fi

        if [ "$task" = "ioi" ]; then
            num_examples_str="--num-examples 1000"
        elif [ "$task" = "mcqa" ]; then
            num_examples_str=""
        else
            num_examples_str="--num-examples 100"
        fi

        python run_attribution.py \
            --models $model \
            --tasks $task \
            --batch-size $batch_size \
            --method $method \
            --ablation $ablation \
            --level $level \
            --ig-steps $ig_steps \
            --pf-gim-filter-quantile $filter_quantile \
            --pf-gim-filter-mode $filter_mode \
            --split train \
            $num_examples_str
    done
done

# Evaluation pass
for model in {interpbench,gpt2,qwen2.5,gemma2,llama3}; do
    for task in {ioi,mcqa,arithmetic_addition,arithmetic_subtraction,arc_easy,arc_challenge}; do
        for absolute in {True,False}; do

            combo="${model}_${task}"
            valid=false
            for valid_combo in "${VALID_COMBOS[@]}"; do
                if [[ "$combo" == "$valid_combo" ]]; then
                    valid=true
                    break
                fi
            done
            if [[ "$valid" == "false" ]]; then
                continue
            fi

            echo "=== Evaluation: $model on $task (absolute: $absolute) ==="

            if [ "$model" = "llama3" ]; then
                batch_size=1
            elif [ "$task" = "arc_easy" ] || [ "$task" = "arc_challenge" ]; then
                batch_size=1
            elif [ "$model" = "gpt2" ] || [ "$model" = "interpbench" ]; then
                batch_size=20
            else
                batch_size=10
            fi

            if [ "$absolute" = "True" ]; then
                absolute_str="--absolute"
            else
                absolute_str=""
            fi

            python run_evaluation.py \
                --models $model \
                --tasks $task \
                --batch-size $batch_size \
                --method $method \
                --ablation $ablation \
                --level $level \
                --split $eval_split \
                $absolute_str
        done
    done
done
