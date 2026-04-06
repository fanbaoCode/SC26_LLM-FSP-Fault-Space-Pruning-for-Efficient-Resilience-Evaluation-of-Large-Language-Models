#!/bin/bash

# 依次运行每个 Python 文件
python ./models_inference_fi/llama2-7b/squad/llama2-7b_fi_neuron_1bit_prefill.py
python ./models_inference_fi/llama2-7b/squad/llama2-7b_fi_neuron_1bit_decode.py
python ./models_inference_fi/llama2-7b/squad/llama2-7b_fi_neuron_2bit_prefill.py
python ./models_inference_fi/llama2-7b/squad/llama2-7b_fi_neuron_2bit_decode.py