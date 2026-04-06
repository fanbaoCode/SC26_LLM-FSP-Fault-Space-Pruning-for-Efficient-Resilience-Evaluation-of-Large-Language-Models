import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json
import logging
import traceback
import random,time
import numpy as np
import datasets
import re
import sys
import os
sys.path.append("/home/fanbao/vscodeWorkplace/pytorchfi")
import pytorchfi.core as pfi_core
import pytorchfi.neuron_error_models as nerr

# 配置日志
logging.basicConfig(
    filename='fault_injection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 加载模型和分词器
local_path = "./models/bert-large"
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForQuestionAnswering.from_pretrained(local_path, torch_dtype=torch.float32)
model = model.to('cuda:0')  # 明确移动到 GPU
model.eval()
device = model.device

# 加载选中的样本
print("从保存的文件加载选中的样本...")
with open("./data/SQuAD2.0/squad_selected_samples.json", "r") as f:
    selected_data = json.load(f)

sample_ids = selected_data["indices"]
selected_samples = datasets.Dataset.from_list(selected_data["samples"])
print(f"已加载 {len(selected_samples)} 个样本用于实验")

# 设置批次大小
batch_size = len(selected_samples)

# 输入和生成函数
def get_batch_inputs(dataset, tokenizer, device, max_length=512):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    references = []
    question_ids = []
    answer_starts = []
    contexts = []
    questions = []
    
    for i in range(len(dataset)):
        context = dataset[i]["context"]
        question = dataset[i]["question"]
        answers = dataset[i]["answers"]
        answer = answers[0]["text"] if answers and isinstance(answers, list) and len(answers) > 0 else ""
        answer_start = answers[0]["answer_start"] if answers and isinstance(answers, list) and len(answers) > 0 else 0
        question_id = dataset[i]["id"]
        
        inputs = tokenizer(
            question,
            context,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        input_ids.append(inputs["input_ids"].squeeze(0))
        attention_masks.append(inputs["attention_mask"].squeeze(0))
        token_type_ids.append(inputs["token_type_ids"].squeeze(0))
        
        references.append(answer)
        question_ids.append(question_id)
        answer_starts.append(answer_start)
        contexts.append(context)
        questions.append(question)
    
    input_ids = torch.stack(input_ids).to(device, non_blocking=True)
    attention_mask = torch.stack(attention_masks).to(device, non_blocking=True)
    token_type_ids = torch.stack(token_type_ids).to(device, non_blocking=True)
    
    return references, input_ids, attention_mask, token_type_ids, question_ids, answer_starts, contexts, questions

def generate_batch(dataset, tokenizer, model, max_length=512):
    references, input_ids, attention_mask, token_type_ids, question_ids, answer_starts, contexts, questions = get_batch_inputs(dataset, tokenizer, device, max_length)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    generated_texts = []
    for i in range(len(dataset)):
        start_logits = outputs.start_logits[i]
        end_logits = outputs.end_logits[i]
        
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()
        
        if end_idx < start_idx:
            end_idx = start_idx
        
        tokens = input_ids[i][start_idx:end_idx + 1]
        answer_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        
        generated_texts.append(answer_text)
    
    del outputs
    del input_ids
    del attention_mask
    del token_type_ids
    torch.cuda.empty_cache()
    
    return generated_texts, references, contexts, question_ids, answer_starts

def faulty_inference_batch(dataset, tokenizer, corrupted_model, max_length=512):
    references, input_ids, attention_mask, token_type_ids, question_ids, answer_starts, contexts, questions = get_batch_inputs(dataset, tokenizer, device, max_length)
    with torch.no_grad():
        outputs = corrupted_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    
    generated_texts = []
    for i in range(len(dataset)):
        start_logits = outputs.start_logits[i]
        end_logits = outputs.end_logits[i]
        
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()
        
        if end_idx < start_idx:
            end_idx = start_idx
        
        tokens = input_ids[i][start_idx:end_idx + 1]
        answer_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        
        generated_texts.append(answer_text)
    
    del outputs
    del input_ids
    del attention_mask
    del token_type_ids
    torch.cuda.empty_cache()
    
    return generated_texts, references, contexts, question_ids, answer_starts

# 实验设置
num_trials = 30000
results = []

# 获取批次信息
references, input_ids, attention_mask, token_type_ids, question_ids, answer_starts, contexts, questions = get_batch_inputs(selected_samples, tokenizer, device)

# 初始化故障注入器
dummy_inputs = tokenizer(
    [selected_samples[i]["question"] for i in range(len(selected_samples))],
    [selected_samples[i]["context"] for i in range(len(selected_samples))],
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
seq_len = dummy_inputs["input_ids"].shape[1]
pfi_model = nerr.single_bit_flip_func(
    model,
    batch_size=batch_size,
    input_shape=[seq_len],
    layer_types=["all"],
    use_cuda=True
)
print(pfi_model.print_pytorchfi_layer_summary())

# 基准推理
baseline_answers, baseline_references, baseline_contexts, baseline_question_ids, baseline_answer_starts = generate_batch(selected_samples, tokenizer, model)

print("=== 基准 (无故障注入) ===")
for idx in range(batch_size):
    print(f"样本 {idx+1}:")
    print(f"问题: {questions[idx]}")
    print(f"基准答案: {baseline_answers[idx]}")
    print(f"参考答案: {baseline_references[idx]}")
    print()

sample_results = []
for idx in range(batch_size):
    sample_results.append({
        "sample_id": idx + 1,
        "context": baseline_contexts[idx],
        "question": questions[idx],
        "ground_truth_answers": selected_samples[idx]["answers"],
        "baseline_output": baseline_answers[idx],
        "faulty_outputs": []
    })

# 故障注入试验
print(f"=== {num_trials} 次随机单比特注入试验（批次处理） ===")
success_trials = 0
while success_trials < num_trials:
    logging.info(f"开始试验 {success_trials + 1}")
    try:
        pfi_model.reset_generate()
        pfi_model.reset_faults()
        corrupted_model = nerr.random_neuron_single_bit_inj_batched(pfi_model)
        faulty_answers, faulty_references, faulty_contexts, faulty_question_ids, faulty_answer_starts = faulty_inference_batch(
            selected_samples, tokenizer, corrupted_model
        )
        fault_infos = pfi_model.last_faults
        print(f"[试验 {success_trials + 1}]")
        for idx in range(batch_size):
            print(f"样本 {idx+1} 答案: {faulty_answers[idx]}")
        print(f"   ↳ 故障信息: {fault_infos}")
        for fault_info in fault_infos:
            batch_idx = fault_info["batch"]
            sample_results[batch_idx]["faulty_outputs"].append({
                "trial": success_trials + 1,
                "output": faulty_answers[batch_idx],
                "fault_info": fault_info
            })

        logging.info(f"试验 {success_trials + 1} 成功完成。故障信息: {pfi_model.last_faults}")
        success_trials += 1

    except Exception as e:
        logging.error(f"试验 {success_trials + 1} 出错: {str(e)}，重新尝试", exc_info=True)

    finally:
        pfi_model.reset_fault_injection()
        if 'corrupted_model' in locals():
            del corrupted_model
        torch.cuda.empty_cache()

results = sample_results
print("\n")

# 保存结果
output_dir = "./results/bert-large/squad"
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/bert-large_fi_neuron_1bit_prefill_FP32.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"实验完成。结果保存至 '{output_dir}/bert-large_fi_neuron_1bit_prefill_FP32.json'。")