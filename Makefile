.PHONY: convert_truthful_qa_to_sft_judger train_judger_info train_judger_truth

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

FOLDER_ROOT:=$(patsubst %/,%,$(MAKEFILE_DIR))
FOLDER_DOWNLOADED_MODEL_ROOT:=$(FOLDER_ROOT)/downloaded_models
FOLDER_TRAINED_MODEL_ROOT:=$(FOLDER_ROOT)/saves/models
FOLDER_LLAMA_FACTORY_DATASET_ROOT:=$(FOLDER_ROOT)/dataset/llama_factory
FOLDER_TRUTHFUL_QA_ROOT:=$(FOLDER_ROOT)/TruthfulQA
FOLDER_HONEST_LLAMA_ROOT:=$(FOLDER_ROOT)/honest_llama
FOLDER_DOLA_ROOT:=$(FOLDER_ROOT)/DoLA
FOLDER_CHAIR_ROOT:=$(FOLDER_ROOT)/chair

FOLDER_PYVENE_ROOT:=$(FOLDER_ROOT)/pyvene
FOLDER_MODEL_PREDICT_ROOT:=$(FOLDER_ROOT)/saves/predict
FOLDER_EXPERIMENT_EVALUATION_ROOT:=$(FOLDER_ROOT)/saves/evaluation
FOLDER_CONTRAST_DISTANCE_AS_PROMPT_RANKER_ROOT:=$(FOLDER_ROOT)/project_contrast_distance_as_prompt_ranker
FOLDER_PROMPT_EVALUATION_ROOT:=$(FOLDER_ROOT)/saves/prompt_evaluation

CONDA_ENV_TRAIN:=eggachecat_llm_train
CONDA_ENV_EVAL:=eggachecat_llm_eval
CONDA_ENV_HONEST_LLAMA:=iti
CONDA_ENV_DP_OPT:=dp-opt
CONDA_ENV_DOLA:=eggachecat_llm_dola

###############################################################################################
# EVALUATION_EXPERIMENT_NAME:=baseline
EVALUATION_EXPERIMENT_NAME:=baseline
EVALUATION_MODEL_KEY:=llama3

EVALUATION_EXPERIMENT_NAME:=wikiqa_full_pay_attention
EVALUATION_MODEL_KEY:=llama3_wikiqa_full_pay_attention

# EVALUATION_EXPERIMENT_NAME:=baseline
# EVALUATION_MODEL_KEY:=llama3
# EVALUATION_EXPERIMENT_NAME:=llama3_8B_instruct_seed_42_top_48_heads_alpha_15/fold_1_com
# EVALUATION_MODEL_KEY:=llama3_8B_instruct
###############################################################################################
FOLDER_EXPERIMENT_EVALUATION:=$(FOLDER_EXPERIMENT_EVALUATION_ROOT)/$(EVALUATION_EXPERIMENT_NAME)


FOLDER_LLAMA_FACTORY:=$(FOLDER_ROOT)/LLaMA-Factory
LLAMA_FACTORY_ENTRY:=$(FOLDER_LLAMA_FACTORY)/src/llamafactory/eggachecat_main.py

HUGGINGFACE_USER:=<YOUR_HUGGINGFACE_USER>
HUGGINGFACE_TOKEN:=<YOUR_HUGGINGFACE_TOKEN>

PYTHONPATH := $(FOLDER_ROOT):$(FOLDER_PYVENE_ROOT):$(FOLDER_TRUTHFUL_QA_ROOT):$(FOLDER_LLAMA_FACTORY):$(FOLDER_LLAMA_FACTORY)/src:$(PYTHONPATH)
export PYTHONPATH
export USE_TORCH=TRUE

dump-env:
	@echo "Makefile directory: $(MAKEFILE_DIR)"
	@echo "FOLDER_ROOT: $(FOLDER_ROOT)"

convert_truthful_qa_to_sft_judger:
	python infra_convert_truthful_qa_to_sft_judger.py

train_judger_info:
	cd $(FOLDER_LLAMA_FACTORY) && \
    CUDA_VISIBLE_DEVICES=0 \
    conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
    python -u $(LLAMA_FACTORY_ENTRY) \
        train \
        --stage sft \
        --do_train \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
        --dataset_dir $(FOLDER_LLAMA_FACTORY_DATASET_ROOT) \
        --dataset truthful_qa_judger_info \
        --template mistral \
        --finetuning_type lora \
        --output_dir $(FOLDER_TRAINED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3/truthful_qa_judger_info/lora/sft \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 1024 \
        --preprocessing_num_workers 16 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --logging_steps 500 \
        --warmup_steps 20 \
        --save_steps 1000 \
        --eval_steps 500 \
        --evaluation_strategy steps \
        --load_best_model_at_end \
        --learning_rate 5e-5 \
        --num_train_epochs 5.0 \
        --val_size 0.1 \
        --plot_loss \
        --fp16

train_judger_truth:
	cd $(FOLDER_LLAMA_FACTORY) && \
    CUDA_VISIBLE_DEVICES=0 \
    conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
    python -u $(LLAMA_FACTORY_ENTRY) \
        train \
        --stage sft \
        --do_train \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
        --dataset_dir $(FOLDER_LLAMA_FACTORY_DATASET_ROOT) \
        --dataset truthful_qa_judger_truth \
        --template mistral \
        --finetuning_type lora \
        --output_dir $(FOLDER_TRAINED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3/truthful_qa_judger_truth/lora/sft \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 1024 \
        --preprocessing_num_workers 16 \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 \
        --lr_scheduler_type cosine \
        --logging_steps 20 \
        --warmup_steps 5 \
        --save_steps 50 \
        --eval_steps 50 \
        --evaluation_strategy steps \
        --load_best_model_at_end \
        --learning_rate 5e-5 \
        --num_train_epochs 5.0 \
        --val_size 0.1 \
        --plot_loss \
        --fp16

serve-model:
	API_PORT=8000 python -u $(LLAMA_FACTORY_ENTRY) api api_serve_llama3.yaml

setup_git_lfs:
	curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
	sudo apt-get install git-lfs
	git lfs install

download_datasets:
	mkdir -p $(FOLDER_DOWNLOADED_MODEL_ROOT)
	cd $(FOLDER_DOWNLOADED_MODEL_ROOT) && \
        git clone https://$(HUGGINGFACE_USER):$(HUGGINGFACE_TOKEN)@huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct && \
        git clone https://$(HUGGINGFACE_USER):$(HUGGINGFACE_TOKEN)@huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 && \
        git clone https://$(HUGGINGFACE_USER):$(HUGGINGFACE_TOKEN)@huggingface.co/huggyllama/llama-7b && \
        git clone https://$(HUGGINGFACE_USER):$(HUGGINGFACE_TOKEN)@huggingface.co/meta-llama/Llama-2-7b-hf && \
        git clone https://$(HUGGINGFACE_USER):$(HUGGINGFACE_TOKEN)@huggingface.co/THUDM/chatglm3-6b

dump-tmux:
	tmux capture-pane -pt $(tmux_id)


init-git:
	git submodule update --init --recursive
	$(MAKE) setup_git_lfs

init-conda-train:
	cd LLaMA-Factory && conda create -y -n $(CONDA_ENV_TRAIN) python=3.10 && \
        conda run --no-capture-output -n $(CONDA_ENV_TRAIN) conda install -y cuda-toolkit pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia/label/cuda-11.8.0 && \
        conda run --no-capture-output -n $(CONDA_ENV_TRAIN) pip install -r requirements.txt

init-conda-eval:
	cd LLaMA-Factory && conda create -y -n $(CONDA_ENV_EVAL) python=3.10 && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) conda install -y cuda-toolkit pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia/label/cuda-11.8.0 && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) pip install -r requirements.txt && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) pip install rouge_score sacrebleu evaluate openai && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) pip install git+https://github.com/google-research/bleurt.git && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) pip install scikit-learn 

init-conda-dp-opt:
	conda create --name $(CONDA_ENV_DP_OPT) python=3.8 -y && \
        conda run --no-capture-output -n $(CONDA_ENV_DP_OPT)  \
                pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118&& && \
        conda run --no-capture-output -n $(CONDA_ENV_DP_OPT)  \
                pip3 transformers==4.28.1 datasets accelerate sentencepiece scikit-learn wandb autodp gradio         
init-conda-dola:
	cd $(FOLDER_DOLA_ROOT) && \
        conda create -y -n $(CONDA_ENV_DOLA) python=3.10 && \
        conda run --no-capture-output -n $(CONDA_ENV_DOLA) \
                conda install -y cuda-toolkit pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia/label/cuda-11.8.0 && \
        cd $(FOLDER_DOLA_ROOT)/transformers-4.28.1 && \
        conda run --no-capture-output -n $(CONDA_ENV_DOLA) \
                pip install -e . && \
        cd $(FOLDER_DOLA_ROOT) \
        conda run --no-capture-output -n $(CONDA_ENV_DOLA) \
                pip install -r requirements.txt

# init=conda-honest_llama:
# 	cd LLaMA-Factory && conda create -y -n $(CONDA_ENV_EVAL) python=3.10 && \


init-conda: init-conda-train init-conda-eval
	echo "inited conda"

# evaluate-model:
# 	python -m truthfulqa.evaluate \
#         --models gpt2 neo-small uqa-small \
#         --metrics mc bleu bleurt \
#         --input_path TruthfulQA_demo.csv \
#         --output_path TruthfulQA_answers.csv \
#         --device 0

evaluate-model-get-response:
	CUDA_VISIBLE_DEVICES=0 cd $(FOLDER_TRUTHFUL_QA_ROOT) && \
    conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
        python \
        -m truthfulqa.evaluate_eggachecat_v2_get_response \
        --input_path TruthfulQA.csv \
        --output_path $(EVALUATION_EXPERIMENT_NAME)/TruthfulQA_answers.csv \
        --tag llama3 \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
        --template llama3 \
        --task eval

evaluate-model-judge-base:
	CUDA_VISIBLE_DEVICES=0 cd $(FOLDER_TRUTHFUL_QA_ROOT) && \
    conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
    python \
        -m truthfulqa.evaluate_eggachecat_v2_judge \
        --input_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers.csv \
        --output_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers_with_metrics_base.csv \
        --metrics mc bleu bleurt \
        --model_key $(EVALUATION_MODEL_KEY)

evaluate-model-judge-info:
	CUDA_VISIBLE_DEVICES=0 cd $(FOLDER_TRUTHFUL_QA_ROOT) && \
    conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
    python \
        -m truthfulqa.evaluate_eggachecat_v2_judge \
        --input_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers_with_metrics_base.csv \
        --output_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers_with_metrics_base_with_info.csv \
        --metrics info \
        --model_key $(EVALUATION_MODEL_KEY) \
        --judge_tag Mistral-7B-Instruct-v0.3-judge-info \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
        --adapter_name_or_path $(FOLDER_TRAINED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3/truthful_qa_judger_info/lora/sft  \

evaluate-model-judge-truth:
	CUDA_VISIBLE_DEVICES=0 cd $(FOLDER_TRUTHFUL_QA_ROOT) && \
    conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
    python \
        -m truthfulqa.evaluate_eggachecat_v2_judge \
        --input_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers_with_metrics_base_with_info.csv \
        --output_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers_with_metrics.csv \
        --metrics judge \
        --model_key $(EVALUATION_MODEL_KEY) \
        --judge_tag Mistral-7B-Instruct-v0.3-judge-truth \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
        --adapter_name_or_path $(FOLDER_TRAINED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3/truthful_qa_judger_truth/lora/sft  \

evaluate-model-judge-summary:
	CUDA_VISIBLE_DEVICES=0 cd $(FOLDER_TRUTHFUL_QA_ROOT) && \
    conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
    python \
        -m truthfulqa.evaluate_eggachecat_v2_summary \
        --input_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers_with_metrics.csv \
        --output_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers_srummary.csv

evaluate-model: evaluate-model-get-response evaluate-model-judge-info evaluate-model-judge-truth evaluate-model-judge-summary
	@echo "doing evaluation"



# --adapter_name_or_path $(FOLDER_TRAINED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3/truthful_qa_judger_info/lora/sft  
predict:
	CUDA_VISIBLE_DEVICES=0 \
        python -u $(LLAMA_FACTORY_ENTRY) train \
        --stage sft \
        --do_predict \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
        --adapter_name_or_path $(FOLDER_TRAINED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3/truthful_qa_judger_info/lora/sft  \
        --eval_dataset alpaca_gpt4_zh,identity \
        --dataset_dir $(FOLDER_LLAMA_FACTORY)/data \
        --template mistral \
        --finetuning_type lora \
        --output_dir ./saves/LLaMA3-8B/lora/predict \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 1024 \
        --preprocessing_num_workers 16 \
        --per_device_eval_batch_size 1 \
        --max_samples 20 \
        --predict_with_generate

honest-llama-get-activation:
	cd $(FOLDER_HONEST_LLAMA_ROOT)/get_activations && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python get_activations.py --model_name llama3_8B_instruct --dataset_name tqa_mc2 && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python get_activations.py --model_name llama3_8B_instruct --dataset_name tqa_gen_end_q

honest-llama-validation:
	cd $(FOLDER_HONEST_LLAMA_ROOT)/validation && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python eggachecat_validate_2fold_get_response.py \
                --model_name llama3_8B_instruct \
                --num_heads 48 \
                --alpha 15 \
                --device 0 \
                --num_fold 2 \
                --use_center_of_mass \
                --instruction_prompt default



project-contrast-distance-as-prompt-kpi-step-1:
	cd $(FOLDER_CONTRAST_DISTANCE_AS_PROMPT_RANKER_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
                python step_1_generate_prompt_given_dataset_description.py

project-contrast-distance-as-prompt-kpi-step-1-post:
	cd $(FOLDER_CONTRAST_DISTANCE_AS_PROMPT_RANKER_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
                python step_1_generate_prompt_given_dataset_description_post.py

project-contrast-distance-as-prompt-kpi-step-2:
	cd $(FOLDER_CONTRAST_DISTANCE_AS_PROMPT_RANKER_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
                python step_2_generate_predicted_prompt_rank_by_contrast_distance.py

project-contrast-distance-as-prompt-kpi-step-3:
	cd $(FOLDER_CONTRAST_DISTANCE_AS_PROMPT_RANKER_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
        CUDA_VISIBLE_DEVICES=0 \
        python step_3_generate_true_prompt_rank_by_running_evaluation.py \
        --cdapr_output_folder $(FOLDER_PROMPT_EVALUATION_ROOT) \
        --cdapr_exp_id baseline_full_size \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
        --template llama3 \
        --task mmlu_test \
        --n_shot 5 \
        --batch_size 5 \
        --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation

# project-contrast-distance-as-prompt-kpi-step-3-baseline:
# 	cd $(FOLDER_LLAMA_FACTORY) && \
#         conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
#         CUDA_VISIBLE_DEVICES=0 python -u $(LLAMA_FACTORY_ENTRY) eval \
#         --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
#         --template llama3 \
#         --task mmlu_test \
#         --lang en \
#         --n_shot 5 \
#         --batch_size 5

# Final MC1/2/3: 
# 0.40758873929008566, 0.5935581847992033, 0.3178663713547437
dola-tqa-mc-baseline:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval.py \
                --model-name $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/output-path-tfmc-baseline.json \
                --num-gpus 1



# Final MC1/2/3: 
# 0.32068543451652387, 0.637717032755801, 0.3204629791533102
dola-tqa-mc-dola-pyvene:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_pyvene.py \
                --model-name $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/output-path-tfmc-dola.json \
                --num-gpus 1 \
                --early-exit-layers 16,18,20,22,24,26,28,30,32

dola-tqa-mc-dola-pyvene-with-weighted-dola:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_pyvene_and_weighted_dola.py \
                --model-name $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/output-path-tfmc-dola.json \
                --num-gpus 1 \
                --early-exit-layers 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32

# dola-tqa-mc-dola-pyvene-with-weighted-dola:
# 	cd $(FOLDER_DOLA_ROOT) && \
#         conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
#         python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_pyvene_and_weighted_dola.py \
#                 --model-name $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
#                 --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
#                 --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/output-path-tfmc-dola.json \
#                 --num-gpus 1 \
#                 --early-exit-layers 16,18,20,22,24,26,28,30,32

# Final MC1/2/3: 
# 0.39167686658506734, 0.6797611143191528, 0.35930669697499584
dola-tqa-mc-dola:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_DOLA) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval.py \
                --model-name huggyllama/llama-7b \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/output-path-tfmc-dola.json \
                --num-gpus 1 \
                --early-exit-layers 16,18,20,22,24,26,28,30,32

run-webchat-after-pay-attention-sft:
	CUDA_VISIBLE_DEVICES=0  conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
        python -u $(LLAMA_FACTORY_ENTRY) \
        webchat \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
        --adapter_name_or_path $(FOLDER_LLAMA_FACTORY)/saves/LLaMA3-8B/lora_pay_attention/sft  \
        --template llama3

run-benchmark-mmlu_test:
	CUDA_VISIBLE_DEVICES=0  conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
        python -u $(LLAMA_FACTORY_ENTRY) \
        eval \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
        --template llama3 \
        --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
        --task mmlu_test \
        --lang en \
        --n_shot 5 \
        --batch_size 1

run-benchmark-mmlu_test-after-pay-attention-sft:
	CUDA_VISIBLE_DEVICES=0  conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
        python -u $(LLAMA_FACTORY_ENTRY) \
        eval \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
        --adapter_name_or_path $(FOLDER_LLAMA_FACTORY)/saves/LLaMA3-8B/lora_pay_attention/sft  \
        --template llama3 \
        --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
        --task mmlu_test \
        --lang en \
        --n_shot 5 \
        --batch_size 1 \

run-sft-only-pay-attention-with-wiki-qa:
	cd $(FOLDER_LLAMA_FACTORY) && \
    CUDA_VISIBLE_DEVICES=0 \
    conda run --no-capture-output -n $(CONDA_ENV_TRAIN) \
    python -u $(LLAMA_FACTORY_ENTRY) \
        train \
        --stage sft \
        --do_train \
        --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
        --dataset_dir $(FOLDER_LLAMA_FACTORY_DATASET_ROOT) \
        --dataset truthful_qa_judger_truth \
        --template llama3 \
        --finetuning_type lora \
        --lora_target_all_except focuse_attention_layer \
        --additional_target focuse_attention_layer \
        --output_dir $(FOLDER_TRAINED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct/wiki_qa/pay_attention/sft \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 1024 \
        --preprocessing_num_workers 16 \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 \
        --lr_scheduler_type cosine \
        --logging_steps 500 \
        --warmup_steps 20 \
        --save_steps 500 \
        --eval_steps 500 \
        --evaluation_strategy steps \
        --load_best_model_at_end \
        --learning_rate 5e-1 \
        --num_train_epochs 5.0 \
        --val_size 0.1 \
        --plot_loss \
        --fp16

evaluate-truthfulqa-model-get-response:
	CUDA_VISIBLE_DEVICES=0 cd $(FOLDER_TRUTHFUL_QA_ROOT) && \
    conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python \
            -m truthfulqa.evaluate_eggachecat_v2_get_response \
            --input_path TruthfulQA.csv \
            --output_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers.csv \
            --tag $(EVALUATION_MODEL_KEY) \
            --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
            --adapter_name_or_path ${FOLDER_ROOT}/saves/Meta-Llama-3-8B-Instruct/wikiqa_full/pay_attention/sft \
            --template llama3 \
            --task eval

evaluate-truthfulqa-judge-base:
	CUDA_VISIBLE_DEVICES=0 cd $(FOLDER_TRUTHFUL_QA_ROOT) && \
    conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
    python \
        -m truthfulqa.evaluate_eggachecat_v2_judge \
        --input_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers.csv \
        --output_path $(FOLDER_EXPERIMENT_EVALUATION)/TruthfulQA_answers_with_metrics_base.csv \
        --model_key $(EVALUATION_MODEL_KEY) \
        --metrics mc bleu bleurt

# 0.3843329253365973 MC2: 0.679487428155513 MC3: 0.35451273532668903
dola-tqa-mc-dola-pyvene-with-baseline:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_llama_factory_and_pyvene.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/dola-tqa-mc-dola-pyvene-with-wiki-qa.json \
                --early-exit-layers 16,18,20,22,24,26,28,30,32 \
                --task eval


dola-tqa-mc-dola-pyvene-with-baseline-layer-selection:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_pyvene.py \
                --model-name $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/output-path-tfmc-dola.json \
                --num-gpus 1 \
                --early-exit-layers 1,2,3,4,5,11,13,16,32

# # Avergaed MC1: 0.390452876376989 MC2: 0.6805199023038555 MC3: 0.3592352975461912
# dola-tqa-mc-dola-pyvene-with-baseline-layer-selection-irrelavant:
# 	cd $(FOLDER_DOLA_ROOT) && \
#         conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
#         python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_pyvene.py \
#                 --model-name $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
#                 --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
#                 --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/output-path-tfmc-dola.json \
#                 --num-gpus 1 \
#                 --early-exit-layers 10,22,24,26,28,32

dola-tqa-mc-dola-pyvene-with-baseline-layer-selection-irrelavant:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_pyvene.py \
                --model-name $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/output-path-tfmc-dola.json \
                --num-gpus 1 \
                --early-exit-layers 5,6,8,15,16,27,32

# --early-exit-layers 0,1,2,3,4,10,12,15,32 \

# 0.3843329253365973 MC2: 0.679487428155513 MC3: 0.35451273532668903
dola-tqa-mc-dola-pyvene-with-wiki-qa:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_llama_factory_and_pyvene.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --adapter_name_or_path ${FOLDER_ROOT}/saves/Meta-Llama-3-8B-Instruct/wikiqa_full/pay_attention/sft \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/dola-tqa-mc-dola-pyvene-with-wiki-qa.json \
                --early-exit-layers 16,18,20,22,24,26,28,30,32 \
                --task eval

dola-tqa-mc-observe-early-decoding-func:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_early_decoding.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-oberservation-layer/Meta-Llama-3-8B-Instruct/layer-$(layer_id).json \
                --early-exit-layers $(layer_id) \
                --task eval

dola-tqa-mc-observe-early-decoding:
	@for i in $(shell seq 1 32); do \
                $(MAKE) dola-tqa-mc-observe-early-decoding-func layer_id=$$i; \
	done

# Num of total question: 2994, correct num: 1197, correct rate: 0.3997995991983968.
dola-factor-eval-baseline:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_with_pyvene.py \
                --model-name $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/output-path-factor-wiki-baseline.json  \
                --num-gpus 1

# Num of total question: 2994, correct num: 1559, correct rate: 0.5207080828323313.
dola-factor-eval-early-exit:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_with_pyvene.py \
                --model-name $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/output-path-factor-wiki-dola-early-exit.json  \
                --early-exit-layers 2,4,6,8,10,12,14,32 \
                --num-gpus 1

# Num of total question: 2994, correct num: 1480, correct rate: 0.49432197728790916.
dola-factor-eval-eggachecat:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_with_eggachecat.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat.json  \
                --template llama3 \
                --task eval
# Num of total question: 2994, correct num: 1350, correct rate: 0.45090180360721444.
dola-factor-eval-eggachecat-per-token:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_with_eggachecat_per_token.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat-per-token.json  \
                --template llama3 \
                --task eval

# Num of total question: 2994, correct num: 1472, correct rate: 0.4916499665998664.
dola-factor-eval-eggachecat-per-token-cross-section-sum:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_with_eggachecat_per_token_cross_section_sum.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat-per-token-cross-section-sum.json  \
                --template llama3 \
                --task eval


dola-factor-eval-eggachecat-per-token-cross-section-mean:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_with_eggachecat_per_token_cross_section_mean.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat-per-token-cross-section-mean.json  \
                --template llama3 \
                --task eval

dola-factor-eval-eggachecat-observe-layers:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_with_eggachecat_observe_layers.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat-observe-layers.json  \
                --template llama3 \
                --task eval

mmlu-few-shot-with-eggachecat-func:
	cd $(FOLDER_LLAMA_FACTORY) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_LLAMA_FACTORY)/src/llamafactory/eggachecat_main_with_eggachecat.py \
                eval \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --template llama3 \
                --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
                --task mmlu_test \
                --lang en \
                --n_shot $(n_shot) \
                --batch_size 1 \
                --save_dir $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/mmlu_test_eggachecat_shot_$(n_shot)

mmlu-few-shot-with-eggachecat:
	@for i in $(shell seq 0 5); do \
                $(MAKE) mmlu-few-shot-with-eggachecat-func n_shot=$$i; \
	done


mmlu-few-shot-baseline-func:
	cd $(FOLDER_LLAMA_FACTORY) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_LLAMA_FACTORY)/src/llamafactory/eggachecat_main.py \
                eval \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --template llama3 \
                --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
                --task mmlu_test \
                --lang en \
                --n_shot $(n_shot) \
                --batch_size 1 \
                --save_dir $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/mmlu_test_baseline_shot_$(n_shot)

mmlu-few-shot-baseline:
	@for i in $(shell seq 0 5); do \
                $(MAKE) mmlu-few-shot-baseline-func n_shot=$$i; \
	done


##############################################################################################################
########## Mistral-7B-Instruct-v0.3
##############################################################################################################
mistral-7B-dola-tqa-mc-observe-early-decoding-func:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_early_decoding.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-oberservation-layer/Mistral-7B-Instruct-v0.3/layer-$(layer_id).json \
                --early-exit-layers $(layer_id) \
                --task eval \
                --template mistral

mistral-7B-dola-tqa-mc-observe-early-decoding:
	@for i in $(shell seq 1 32); do \
                $(MAKE) mistral-7B-dola-tqa-mc-observe-early-decoding-func layer_id=$$i; \
	done

mistral-7B-dola-tqa-mc-dola-pyvene-with-baseline:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_llama_factory_and_pyvene.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/Mistral-7B-Instruct-v0.3/dola-tqa-mc.json \
                --early-exit-layers 16,18,20,22,24,26,28,30,32 \
                --task eval \
                --template mistral

mistral-7B-dola-factor-eval-eggachecat-observe-layers:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_baseline_observe_layers.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/Mistral-7B-Instruct-v0.3/factor-wiki-dola-eggachecat-observe-layers.json  \
                --template mistral \
                --task eval

mistral-7B-dola-factor-eval-dola:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_with_pyvene_and_llama_factory.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --template mistral \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/Mistral-7B-Instruct-v0.3/factor-wiki-dola.json  \
                --early-exit-layers 2,4,6,8,10,12,14,32 \
                --task eval

mistral-7B-mmlu-few-shot-with-eggachecat-func:
	cd $(FOLDER_LLAMA_FACTORY) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_LLAMA_FACTORY)/src/llamafactory/eggachecat_main_with_eggachecat.py \
                eval \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --template mistral \
                --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
                --task mmlu_test \
                --lang en \
                --n_shot $(n_shot) \
                --batch_size 1 \
                --save_dir $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/Mistral-7B-Instruct-v0.3/mmlu_test_eggachecat_shot_$(n_shot)

mistral-7B-mmlu-few-shot-with-eggachecat:
	@for i in $(shell seq 0 1); do \
                $(MAKE) mistral-7B-mmlu-few-shot-with-eggachecat-func n_shot=$$i; \
	done

mistral-7B-mmlu-few-shot-baseline-func:
	cd $(FOLDER_LLAMA_FACTORY) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_LLAMA_FACTORY)/src/llamafactory/eggachecat_main.py \
                eval \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --template mistral \
                --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
                --task mmlu_test \
                --lang en \
                --n_shot $(n_shot) \
                --batch_size 1 \
                --save_dir $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/Mistral-7B-Instruct-v0.3/mmlu_test_baseline_shot_$(n_shot)

mistral-7B-mmlu-few-shot-baseline:
	@for i in $(shell seq 0 1); do \
                $(MAKE) mistral-7B-mmlu-few-shot-baseline-func n_shot=$$i; \
	done

##############################################################################################################
########## Mistral-7B-Instruct-v0.3
##############################################################################################################
Mistral-7B-Instruct-v0.3-dola-tqa-mc-observe-early-decoding-func:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_early_decoding.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-oberservation-layer/Mistral-7B-Instruct-v0.3/layer-$(layer_id).json \
                --early-exit-layers $(layer_id) \
                --task eval \
                --template mistral

Mistral-7B-Instruct-v0.3-dola-tqa-mc-observe-early-decoding:
	@for i in $(shell seq 1 32); do \
                $(MAKE) Mistral-7B-Instruct-v0.3-dola-tqa-mc-observe-early-decoding-func layer_id=$$i; \
	done

Mistral-7B-Instruct-v0.3-dola-tqa-mc-dola-pyvene-with-baseline:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/tfqa_mc_eval_with_llama_factory_and_pyvene.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/dola-baseline/Mistral-7B-Instruct-v0.3/dola-tqa-mc.json \
                --early-exit-layers 16,18,20,22,24,26,28,30,32 \
                --task eval \
                --template mistral

Mistral-7B-Instruct-v0.3-dola-factor-eval-eggachecat-observe-layers:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_baseline_observe_layers.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/Mistral-7B-Instruct-v0.3/factor-wiki-dola-eggachecat-observe-layers.json  \
                --template mistral \
                --task eval

Mistral-7B-Instruct-v0.3-dola-factor-eval-dola:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_with_pyvene_and_llama_factory.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --template mistral \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/Mistral-7B-Instruct-v0.3/factor-wiki-dola.json  \
                --early-exit-layers 2,4,6,8,10,12,14,32 \
                --task eval

Mistral-7B-Instruct-v0.3-mmlu-few-shot-with-eggachecat-func:
	cd $(FOLDER_LLAMA_FACTORY) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_LLAMA_FACTORY)/src/llamafactory/eggachecat_main_with_eggachecat.py \
                eval \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --template mistral \
                --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
                --task mmlu_test \
                --lang en \
                --n_shot $(n_shot) \
                --batch_size 1 \
                --save_dir $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/Mistral-7B-Instruct-v0.3/mmlu_test_eggachecat_shot_$(n_shot)

Mistral-7B-Instruct-v0.3-mmlu-few-shot-with-eggachecat:
	@for i in $(shell seq 0 1); do \
                $(MAKE) Mistral-7B-Instruct-v0.3-mmlu-few-shot-with-eggachecat-func n_shot=$$i; \
	done

Mistral-7B-Instruct-v0.3-mmlu-few-shot-baseline-func:
	cd $(FOLDER_LLAMA_FACTORY) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_LLAMA_FACTORY)/src/llamafactory/eggachecat_main.py \
                eval \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --template mistral \
                --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
                --task mmlu_test \
                --lang en \
                --n_shot $(n_shot) \
                --batch_size 1 \
                --save_dir $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/Mistral-7B-Instruct-v0.3/mmlu_test_baseline_shot_$(n_shot)

Mistral-7B-Instruct-v0.3-mmlu-few-shot-baseline:
	@for i in $(shell seq 0 1); do \
                $(MAKE) Mistral-7B-Instruct-v0.3-mmlu-few-shot-baseline-func n_shot=$$i; \
	done

Mistral-7B-Instruct-v0.3-run:
	$(MAKE) Mistral-7B-Instruct-v0.3-dola-tqa-mc-observe-early-decoding && \
        $(MAKE) Mistral-7B-Instruct-v0.3-dola-tqa-mc-dola-pyvene-with-baseline  && \
        $(MAKE) Mistral-7B-Instruct-v0.3-dola-factor-eval-eggachecat-observe-layers  && \
        $(MAKE) Mistral-7B-Instruct-v0.3-dola-factor-eval-dola && \
        $(MAKE) Mistral-7B-Instruct-v0.3-mmlu-few-shot-with-eggachecat && \
        $(MAKE) Mistral-7B-Instruct-v0.3-mmlu-few-shot-baseline

###### just for test
llama-3-dola-factor-eval-eggachecat-observe-layers:
	cd $(FOLDER_DOLA_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_DOLA_ROOT)/factor_eval_baseline_observe_layers.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/factor_eval/wiki_factor/Meta-Llama-3-8B-Instruct/factor-wiki-dola-eggachecat-observe-layers.json  \
                --template llama3 \
                --task eval
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
#################################################################
######################### tfqa_mc_eval
#################################################################
chair-baseline_and_observe_layers-tfqa_mc_eval-llama-7b:
	cd $(FOLDER_CHAIR_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observer_layers_tfqa_mc_eval.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/llama-7b \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/truthfulqa/llama-7b/baseline_and_observe_layers_tfqa_mc_eval.json  \
                --template llama3 \
                --task eval

chair-baseline_and_observe_layers-tfqa_mc_eval-Meta-Llama-3-8B-Instruct:
	cd $(FOLDER_CHAIR_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observer_layers_tfqa_mc_eval.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/truthfulqa/Meta-Llama-3-8B-Instruct/baseline_and_observe_layers_tfqa_mc_eval.json  \
                --template llama3 \
                --task eval

chair-baseline_and_observe_layers-tfqa_mc_eval-Llama-2-7b-hf:
	cd $(FOLDER_CHAIR_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observer_layers_tfqa_mc_eval.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Llama-2-7b-hf \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/truthfulqa/Llama-2-7b-hf/baseline_and_observe_layers_tfqa_mc_eval.json  \
                --template llama2 \
                --task eval

chair-baseline_and_observe_layers-tfqa_mc_eval-Mistral-7B-Instruct-v0.3:
	cd $(FOLDER_CHAIR_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observer_layers_tfqa_mc_eval.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --data-path $(FOLDER_TRUTHFUL_QA_ROOT) \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/truthfulqa/Mistral-7B-Instruct-v0.3/baseline_and_observe_layers_tfqa_mc_eval.json  \
                --template mistral \
                --task eval
#################################################################
######################### factor_eval
#################################################################
chair-baseline_and_observe_layers-factor_eval-llama-7b:
	cd $(FOLDER_CHAIR_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observe_layers_factor_eval.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/llama-7b \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/factor_eval/wiki_factor/llama-7b/baseline_and_observe_layers_factor_eval.json  \
                --template llama3 \
                --task eval

chair-baseline_and_observe_layers-factor_eval-Meta-Llama-3-8B-Instruct:
	cd $(FOLDER_CHAIR_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observe_layers_factor_eval.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/factor_eval/wiki_factor/Meta-Llama-3-8B-Instruct/baseline_and_observe_layers_factor_eval.json  \
                --template llama3 \
                --task eval

chair-baseline_and_observe_layers-factor_eval-Llama-2-7b-hf:
	cd $(FOLDER_CHAIR_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observe_layers_factor_eval.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Llama-2-7b-hf \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/factor_eval/wiki_factor/Llama-2-7b-hf/baseline_and_observe_layers_factor_eval.json  \
                --template llama2 \
                --task eval

chair-baseline_and_observe_layers-factor_eval-Mistral-7B-Instruct-v0.3:
	cd $(FOLDER_CHAIR_ROOT) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observe_layers_factor_eval.py \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --data-path $(FOLDER_ROOT)/saves/wiki_factor/wiki_factor.csv \
                --output-path $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/factor_eval/wiki_factor/Mistral-7B-Instruct-v0.3/baseline_and_observe_layers_factor_eval.json  \
                --template mistral \
                --task eval
#################################################################
######################### mmlu_test
#################################################################
N_FEW_SHOT_LIST := 0 1 5

chair-baseline_and_observe_layers-mmlu_test-few_shot-Meta-Llama-3-8B-Instruct-func:
	cd $(FOLDER_LLAMA_FACTORY) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observe_layers_llama_factory.py \
                eval \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Meta-Llama-3-8B-Instruct \
                --template llama3 \
                --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
                --task mmlu_test \
                --lang en \
                --n_shot $(n_shot) \
                --batch_size 1 \
                --save_dir $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/mmlu_test/Meta-Llama-3-8B-Instruct/mmlu_test_baseline_shot_$(n_shot)

chair-baseline_and_observe_layers-mmlu_test-few_shot-Meta-Llama-3-8B-Instruct:
	@for i in $(N_FEW_SHOT_LIST); do \
                $(MAKE) chair-baseline_and_observe_layers-mmlu_test-few_shot-Meta-Llama-3-8B-Instruct-func n_shot=$$i; \
	done


chair-baseline_and_observe_layers-mmlu_test-few_shot-Mistral-7B-Instruct-v0.3-func:
	cd $(FOLDER_LLAMA_FACTORY) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observe_layers_llama_factory.py \
                eval \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --template mistral \
                --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
                --task mmlu_test \
                --lang en \
                --n_shot $(n_shot) \
                --batch_size 1 \
                --save_dir $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/mmlu_test/Mistral-7B-Instruct-v0.3/mmlu_test_baseline_shot_$(n_shot)

chair-baseline_and_observe_layers-mmlu_test-few_shot-Mistral-7B-Instruct-v0.3:
	@for i in $(N_FEW_SHOT_LIST); do \
                $(MAKE) chair-baseline_and_observe_layers-mmlu_test-few_shot-Mistral-7B-Instruct-v0.3-func n_shot=$$i; \
	done

chair-baseline_and_observe_layers-mmlu_test-few_shot-Llama-2-7b-hf-func:
	cd $(FOLDER_LLAMA_FACTORY) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observe_layers_llama_factory.py \
                eval \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Llama-2-7b-hf \
                --template llama2 \
                --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
                --task mmlu_test \
                --lang en \
                --n_shot $(n_shot) \
                --batch_size 1 \
                --save_dir $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/mmlu_test/Llama-2-7b-hf/mmlu_test_baseline_shot_$(n_shot)

chair-baseline_and_observe_layers-mmlu_test-few_shot-Llama-2-7b-hf:
	@for i in $(N_FEW_SHOT_LIST); do \
                $(MAKE) chair-baseline_and_observe_layers-mmlu_test-few_shot-Llama-2-7b-hf-func n_shot=$$i; \
	done

chair-baseline_and_observe_layers-mmlu_test-few_shot-Mistral-7B-Instruct-v0.3-func:
	cd $(FOLDER_LLAMA_FACTORY) && \
        conda run --no-capture-output -n $(CONDA_ENV_EVAL) \
        python $(FOLDER_CHAIR_ROOT)/baseline_and_observe_layers_llama_factory.py \
                eval \
                --model_name_or_path $(FOLDER_DOWNLOADED_MODEL_ROOT)/Mistral-7B-Instruct-v0.3 \
                --template mistral \
                --task_dir $(FOLDER_LLAMA_FACTORY)/evaluation \
                --task mmlu_test \
                --lang en \
                --n_shot $(n_shot) \
                --batch_size 1 \
                --save_dir $(FOLDER_EXPERIMENT_EVALUATION_ROOT)/chair/mmlu_test/Mistral-7B-Instruct-v0.3/mmlu_test_baseline_shot_$(n_shot)

chair-baseline_and_observe_layers-mmlu_test-few_shot-Mistral-7B-Instruct-v0.3:
	@for i in $(N_FEW_SHOT_LIST); do \
                $(MAKE) chair-baseline_and_observe_layers-mmlu_test-few_shot-Mistral-7B-Instruct-v0.3-func n_shot=$$i; \
	done