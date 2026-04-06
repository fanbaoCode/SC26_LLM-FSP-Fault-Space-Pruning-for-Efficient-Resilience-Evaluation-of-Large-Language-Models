#!/bin/bash

# bart-large模型在lambada数据集上的实验
python ./models_inference_fi/bart-large/lambada/bart-large_fi_neuron_1bit_prefill.py
python ./models_inference_fi/bart-large/lambada/bart-large_fi_neuron_2bit_prefill.py

#bert-large模型在squad数据集上的实验
python ./models_inference_fi/bert-large/squad/bert-large_fi_neuron_1bit_prefill.py
python ./models_inference_fi/bert-large/squad/bert-large_fi_neuron_2bit_prefill.py

# roberta-large模型在MNLI数据集上的实验
python ./models_inference_fi/roberta-large/MNLI/roberta-large_fi_neuron_1bit_prefill.py
python ./models_inference_fi/roberta-large/MNLI/roberta-large_fi_neuron_2bit_prefill.py

# t5-3b模型在lambada数据集上的实验
python ./models_inference_fi/t5-3b/lambada/t5-3b_fi_neuron_1bit_prefill.py
python ./models_inference_fi/t5-3b/lambada/t5-3b_fi_neuron_2bit_prefill.py

# t5-3b模型在squad数据集上的实验
python ./models_inference_fi/t5-3b/squad/t5-3b_fi_neuron_1bit_prefill.py
python ./models_inference_fi/t5-3b/squad/t5-3b_fi_neuron_1bit_decode.py
python ./models_inference_fi/t5-3b/squad/t5-3b_fi_neuron_2bit_prefill.py
python ./models_inference_fi/t5-3b/squad/t5-3b_fi_neuron_2bit_decode.py

# qwen2-7b模型在squad数据集上的实验
python ./models_inference_fi/qwen2-7b/squad/qwen2-7b_fi_neuron_1bit_prefill.py
python ./models_inference_fi/qwen2-7b/squad/qwen2-7b_fi_neuron_1bit_decode.py
python ./models_inference_fi/qwen2-7b/squad/qwen2-7b_fi_neuron_2bit_prefill.py
python ./models_inference_fi/qwen2-7b/squad/qwen2-7b_fi_neuron_2bit_decode.py

# qwen2-7b模型在gsm8k数据集上的实验
python ./models_inference_fi/qwen2-7b/gsm8k/qwen2-7b_fi_neuron_1bit_prefill.py
python ./models_inference_fi/qwen2-7b/gsm8k/qwen2-7b_fi_neuron_1bit_decode.py
python ./models_inference_fi/qwen2-7b/gsm8k/qwen2-7b_fi_neuron_2bit_prefill.py
python ./models_inference_fi/qwen2-7b/gsm8k/qwen2-7b_fi_neuron_2bit_decode.py


# llama2-7b模型在squad数据集上的实验
python ./models_inference_fi/llama2-7b/squad/llama2-7b_fi_neuron_1bit_prefill.py
python ./models_inference_fi/llama2-7b/squad/llama2-7b_fi_neuron_1bit_decode.py
python ./models_inference_fi/llama2-7b/squad/llama2-7b_fi_neuron_2bit_prefill.py
python ./models_inference_fi/llama2-7b/squad/llama2-7b_fi_neuron_2bit_decode.py

# llama2-7b模型在gsm8k数据集上的实验
python ./models_inference_fi/llama2-7b/gsm8k/llama2-7b_fi_neuron_1bit_prefill.py
python ./models_inference_fi/llama2-7b/gsm8k/llama2-7b_fi_neuron_1bit_decode.py
python ./models_inference_fi/llama2-7b/gsm8k/llama2-7b_fi_neuron_2bit_prefill.py
python ./models_inference_fi/llama2-7b/gsm8k/llama2-7b_fi_neuron_2bit_decode.py
