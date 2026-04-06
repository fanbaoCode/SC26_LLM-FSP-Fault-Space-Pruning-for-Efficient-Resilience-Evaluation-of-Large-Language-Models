import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json, time
import logging
import traceback
import numpy as np
import re
import sys
import string
sys.path.append("/home/fanbao/vscodeWorkplace/pytorchfi")
import pytorchfi.core as pfi_core
import pytorchfi.neuron_error_models as nerr
import os

# 配置日志
logging.basicConfig(
    filename='fault_injection_lambada.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 加载模型和分词器
local_path = "./models/bart-large"
tokenizer = AutoTokenizer.from_pretrained(local_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'left'
model = AutoModelForSeq2SeqLM.from_pretrained(local_path, torch_dtype=torch.float32)
model = model.to('cuda:0')
model.eval()
device = model.device

# 加载选定的 LAMBADA 数据集
print("加载选定的 lambada 数据集...")
with open("./data/lambada/lambada_selected_samples.json", "r", encoding="utf-8") as f:
    selected_samples = json.load(f)
print(f"已加载 {len(selected_samples)} 个样本用于实验")

# 设置批次大小
batch_size = len(selected_samples)  # 与 select_data_lambada.py 一致

# 输入和生成函数（批次处理）
def get_last_non_punct_word(seq):
    tokens = seq.strip().split()
    for token in reversed(tokens):
        if not all(ch in string.punctuation for ch in token):
            return token
    return ""

def get_batch_inputs(dataset, tokenizer, device):
    """为整个批次的 LAMBADA 数据集准备输入"""
    prompts = []
    references = []
    sample_ids = []
    
    for i, sample in enumerate(dataset):
        prompt = sample["prompt"].strip()
        label = sample["label"].strip()
        sample_id = sample.get("id", f"sample_{i}")
        
        prompts.append(prompt)
        references.append(label)
        sample_ids.append(sample_id)
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device, non_blocking=True)
    attention_mask = inputs["attention_mask"].to(device, non_blocking=True)
    
    return references, prompts, input_ids, attention_mask, sample_ids

def generate_batch(dataset, tokenizer, model, max_new_tokens=5):
    """使用贪婪搜索生成批次 LAMBADA 样本的预测"""
    references, prompts, input_ids, attention_mask, sample_ids = get_batch_inputs(dataset, tokenizer, device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,
            top_p=None,
            top_k=None
        )
    generated_texts = []
    for i in range(len(prompts)):
        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
        last_word = get_last_non_punct_word(generated_text)
        generated_texts.append(last_word)
    
    # 清理内存
    del outputs
    del input_ids
    del attention_mask
    torch.cuda.empty_cache()
    
    return generated_texts, references, prompts, sample_ids

# 故障注入推理函数（批次处理）
def faulty_inference_batch(dataset, tokenizer, corrupted_model, max_new_tokens=5):
    references, prompts, input_ids, attention_mask, sample_ids = get_batch_inputs(dataset, tokenizer, device)
    
    with torch.no_grad():
        outputs = corrupted_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,
            top_p=None,
            top_k=None
        )
    
    generated_texts = []
    for i in range(len(prompts)):
        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
        last_word = get_last_non_punct_word(generated_text)
        generated_texts.append(last_word)
    
    del outputs
    del input_ids
    del attention_mask
    torch.cuda.empty_cache()
    
    return generated_texts, references, prompts, sample_ids

# 实验设置
num_trials = 30000  # 总试验次数
results = []

# 获取批次信息
references, prompts, _, _, sample_ids = get_batch_inputs(selected_samples, tokenizer, device)

# 初始化故障注入器
dummy_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
seq_len = dummy_inputs["input_ids"].shape[1]

# 临时修改 forward 方法以提供 dummy decoder_input_ids
original_forward = model.forward
def dummy_forward(*args, **kwargs):
    if args:
        if len(args) > 0:
            kwargs['input_ids'] = args[0]
        if len(args) > 1:
            kwargs['decoder_input_ids'] = args[1]
    
    if 'input_ids' in kwargs and kwargs['input_ids'] is not None:
        input_shape = kwargs['input_ids'].shape
        kwargs['input_ids'] = torch.randint(0, model.config.vocab_size, input_shape, dtype=torch.long, device=device)
    
    if 'decoder_input_ids' not in kwargs or kwargs['decoder_input_ids'] is None:
        batch_size = kwargs['input_ids'].shape[0]
        kwargs['decoder_input_ids'] = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    
    for key in kwargs:
        if isinstance(kwargs[key], torch.Tensor) and kwargs[key].device != device:
            kwargs[key] = kwargs[key].to(device)
    
    return original_forward(**kwargs)

model.forward = dummy_forward

# 初始化 PyTorchFI
try:
    pfi_model = nerr.single_bit_flip_func(
        model,
        batch_size=batch_size,
        input_shape=[seq_len],
        layer_types=["all"],
        use_cuda=True
    )
except Exception as e:
    logging.error(f"PyTorchFI 初始化失败: {str(e)}\n{traceback.format_exc()}")
    print(f"PyTorchFI 初始化失败: {str(e)}")
    sys.exit(1)

# 恢复原始 forward 方法
model.forward = original_forward

print(pfi_model.print_pytorchfi_layer_summary())

# 基准推理（批次处理）
try:
    baseline_answers, baseline_references, baseline_prompts, baseline_sample_ids = generate_batch(selected_samples, tokenizer, model)
except Exception as e:
    logging.error(f"基准推理失败: {str(e)}\n{traceback.format_exc()}")
    print(f"基准推理失败: {str(e)}")
    sys.exit(1)

print("=== 基准 (无故障注入) ===")
for idx in range(len(baseline_prompts)):
    print(f"样本 {idx+1}:")
    print(f"提示: {baseline_prompts[idx][:50]}...")
    print(f"基准预测: {baseline_answers[idx]}")
    print(f"参考答案: {baseline_references[idx]}")
    print()

sample_results = []
for idx in range(len(baseline_prompts)):
    sample_results.append({
        "sample_id": idx + 1,
        "prompt": baseline_prompts[idx],
        "ground_truth": baseline_references[idx],
        "baseline_output": baseline_answers[idx],
        "faulty_outputs": []
    })

# 故障注入试验（批次处理）
print(f"=== {num_trials} 次随机单比特注入试验（批次处理） ===")
for i in range(num_trials):
    logging.info(f"开始试验 {i + 1}")
    pfi_model.reset_generate()
    pfi_model.reset_faults()
    corrupted_model = nerr.random_neuron_single_bit_inj_batched(pfi_model)
    faulty_answers, faulty_references, faulty_prompts, faulty_sample_ids = faulty_inference_batch(selected_samples, tokenizer, corrupted_model)
    fault_infos = pfi_model.last_faults
    
    print(f"[试验 {i+1}]")
    for idx in range(len(faulty_prompts)):
        print(f"样本 {idx+1} 预测: {faulty_answers[idx]}")
    print(f"   ↳ 故障信息: {fault_infos}")
    
    for fault_info in fault_infos:
        batch_idx = fault_info["batch"]
        sample_results[batch_idx]["faulty_outputs"].append({
            "trial": i + 1,
            "output": faulty_answers[batch_idx],
            "fault_info": fault_info
        })
    logging.info(f"试验 {i + 1} 成功完成。故障信息: {pfi_model.last_faults}")
    pfi_model.reset_fault_injection()
    del corrupted_model
    torch.cuda.empty_cache()

results = sample_results
print("\n")

# 保存结果
output_dir = "./results/bart-large/lambada"
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/bart-large_fi_neuron_1bit_prefill_FP32.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"实验完成。结果保存至 '{output_dir}/bart-large_fi_neuron_1bit_prefill_FP32.json'。")