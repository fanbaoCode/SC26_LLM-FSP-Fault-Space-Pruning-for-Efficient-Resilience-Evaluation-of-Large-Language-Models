import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging
import traceback
import random
import numpy as np
import datasets
import re
import sys
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
local_path = "./models/llama2-7b"
tokenizer = AutoTokenizer.from_pretrained(local_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
device = model.device

# 从保存的 JSON 文件加载选中的样本
print("从保存的文件加载选中的样本...")
with open("./data/SQuAD2.0/squad_selected_samples.json", "r") as f:
    selected_data = json.load(f)

sample_ids = selected_data["indices"]
selected_samples = datasets.Dataset.from_list(selected_data["samples"])
print(f"已加载 {len(selected_samples)} 个样本用于实验")

# 设置批次大小为样本数量（这里假设为5）
batch_size = len(selected_samples)

# 输入和生成函数（修改为批次处理）
def get_batch_inputs(dataset, tokenizer, device):
    """为整个批次的 SQuAD 数据集准备输入"""
    prompts = []
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
        
        prompt = f"Answer the question based on the following context:\nContext: {context}\nQuestion: {question}\nAnswer:"
        
        prompts.append(prompt)
        references.append(answer)
        question_ids.append(question_id)
        answer_starts.append(answer_start)
        contexts.append(context)
        questions.append(question)
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device, non_blocking=True)
    attention_mask = inputs["attention_mask"].to(device, non_blocking=True)
    
    return references, prompts, input_ids, attention_mask, question_ids, answer_starts, contexts, questions

def generate_batch(dataset, tokenizer, model, max_new_tokens=50):
    """使用贪婪搜索生成批次 SQuAD 样本的答案"""
    references, prompts, input_ids, attention_mask, question_ids, answer_starts, contexts, questions = get_batch_inputs(dataset, tokenizer, device)
    prompt_lens = [len(ids) for ids in input_ids]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max(prompt_lens) + max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,
            top_p=None,
            top_k=None
        )
    
    generated_texts = []
    for i in range(batch_size):
        generated_text = tokenizer.decode(outputs[i][prompt_lens[i]:], skip_special_tokens=True)
        
        # 处理生成的答案：提取第一句话
        if "\n" in generated_text:
            generated_text = generated_text.split("\n")[0].strip()
        
        first_sentence = ""
        sentence_endings = re.split(r'(?<=[.!?])\s+', generated_text)
        if sentence_endings and len(sentence_endings) > 0:
            first_sentence = sentence_endings[0].strip()
        else:
            first_sentence = generated_text.strip()
        
        generated_texts.append(first_sentence)
    
    # 清理内存
    del outputs
    del input_ids
    del attention_mask
    torch.cuda.empty_cache()
    
    return generated_texts, references, contexts, question_ids, answer_starts

# 故障注入推理函数（修改为批次处理）
def faulty_inference_batch(dataset, tokenizer, corrupted_model, max_new_tokens=50):
    references, prompts, input_ids, attention_mask, question_ids, answer_starts, contexts, questions = get_batch_inputs(dataset, tokenizer, device)
    prompt_lens = [len(ids) for ids in input_ids]
    
    with torch.no_grad():
        outputs = corrupted_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max(prompt_lens) + max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,
            top_p=None,
            top_k=None
        )
    
    generated_texts = []
    for i in range(batch_size):
        generated_text = tokenizer.decode(outputs[i][prompt_lens[i]:], skip_special_tokens=True)
        
        if "\n" in generated_text:
            generated_text = generated_text.split("\n")[0].strip()
        
        first_sentence = ""
        sentence_endings = re.split(r'(?<=[.!?])\s+', generated_text)
        if sentence_endings and len(sentence_endings) > 0:
            first_sentence = sentence_endings[0].strip()
        else:
            first_sentence = generated_text.strip()
        
        generated_texts.append(first_sentence)
    
    del outputs
    del input_ids
    del attention_mask
    torch.cuda.empty_cache()
    
    return generated_texts, references, contexts, question_ids, answer_starts

# 实验设置
num_trials = 15000  # 总试验次数
results = []
# 设置批次大小为样本数量的一半
batch_size = len(selected_samples) // 2
all_sample_results = []

for batch_idx in range(2):
    start = batch_idx * batch_size
    end = (batch_idx + 1) * batch_size if batch_idx < 1 else len(selected_samples)
    batch_samples = selected_samples.select(range(start, end))

    # 获取批次信息
    references, prompts, _, _, question_ids, answer_starts, contexts, questions = get_batch_inputs(batch_samples, tokenizer, device)

    # 初始化故障注入器
    dummy_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    seq_len = dummy_inputs["input_ids"].shape[1]
    pfi_model = nerr.single_bit_flip_func(
        model,
        batch_size=(end - start),
        input_shape=[seq_len],
        layer_types=["all"],
        use_cuda=True
    )
    print(pfi_model.print_pytorchfi_layer_summary())

    # 基准推理
    baseline_answers, baseline_references, baseline_contexts, baseline_question_ids, baseline_answer_starts = generate_batch(batch_samples, tokenizer, model)

    print("=== 基准 (无故障注入) ===")
    for idx in range(end - start):
        print(f"样本 {start + idx + 1}:")
        print(f"问题: {questions[idx]}")
        print(f"基准答案: {baseline_answers[idx]}")
        print(f"参考答案: {baseline_references[idx]}")
        print()

    sample_results = []
    for idx in range(end - start):
        sample_results.append({
            "sample_id": start + idx + 1,
            "context": baseline_contexts[idx],
            "question": questions[idx],
            "ground_truth_answers": batch_samples[idx]["answers"],
            "baseline_output": baseline_answers[idx],
            "faulty_outputs": []
        })

    # 故障注入试验
    print(f"=== {num_trials} 次随机单比特注入试验（批次处理） ===")
    for i in range(num_trials):
        try:
            logging.info(f"开始试验 {i + 1}")
            pfi_model.reset_generate()
            pfi_model.reset_faults()
            corrupted_model = nerr.random_neuron_two_bit_inj_batched(pfi_model)
            faulty_answers, faulty_references, faulty_contexts, faulty_question_ids, faulty_answer_starts = faulty_inference_batch(batch_samples, tokenizer, corrupted_model)
            fault_infos = pfi_model.last_faults

            print(f"[试验 {i+1}]")
            for idx in range(end - start):
                print(f"样本 {start + idx + 1} 答案: {faulty_answers[idx]}")
            print(f"   ↳ 故障信息: {fault_infos}")

            for fault_info in fault_infos:
                batch_idx_in_batch = fault_info["batch"]
                sample_results[batch_idx_in_batch]["faulty_outputs"].append({
                    "trial": i + 1,
                    "output": faulty_answers[batch_idx_in_batch],
                    "fault_info": fault_info
                })
            logging.info(f"试验 {i + 1} 成功完成。故障信息: {pfi_model.last_faults}")
        except Exception as e:
            i -= 1
            continue
        pfi_model.reset_fault_injection()
        del corrupted_model
        torch.cuda.empty_cache()

    # 合并每个批次的结果
    all_sample_results.extend(sample_results)

print("\n")

# 保存所有批次合并后的结果
output_dir = "./results/llama2-7b/squad"
with open(f"{output_dir}/llama2-7b_fi_neuron_2bit_prefill_FP32.json", "w") as f:
    json.dump(all_sample_results, f, indent=4)

print(f"实验完成。结果保存至 '{output_dir}/llama2-7b_fi_neuron_2bit_prefill_FP32.json'。")