import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import json
import logging
import traceback
import numpy as np
import datasets
import re
import sys
sys.path.append("/home/fanbao/vscodeWorkplace/pytorchfi")
import pytorchfi.core as pfi_core
import pytorchfi.neuron_error_models as nerr
import time

# 配置日志
logging.basicConfig(
    filename='fault_injection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 加载模型和分词器
local_path = "./models/t5-3b"
tokenizer = T5Tokenizer.from_pretrained(local_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'left'
model = AutoModelForSeq2SeqLM.from_pretrained(local_path, torch_dtype=torch.float32)
model = model.to('cuda:0')
model.eval()
device = model.device

# 从保存的 JSON 文件加载选中的样本
print("从保存的文件加载选中的样本...")
with open("./data/SQuAD2.0/squad_selected_samples.json", "r") as f:
    selected_data = json.load(f)

sample_ids = selected_data["indices"]
selected_samples = datasets.Dataset.from_list(selected_data["samples"])
print(f"已加载 {len(selected_samples)} 个样本用于实验")

# 设置批次大小为样本数量
batch_size = len(selected_samples)

# 输入和生成函数（批次处理）
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
        
        # 使用标准 T5 QA 提示格式，参考你的样本选择脚本
        prompt = f"question: {question} context: {context}"
        
        prompts.append(prompt)
        references.append(answer)
        question_ids.append(question_id)
        answer_starts.append(answer_start)
        contexts.append(context)
        questions.append(question)
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
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
    for i in range(batch_size):
        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        
        if "\n" in generated_text:
            generated_text = generated_text.split("\n")[0].strip()
        
        sentence_endings = re.split(r'(?<=[.!?])\s+', generated_text)
        first_sentence = sentence_endings[0].strip() if sentence_endings else generated_text.strip()
        generated_texts.append(first_sentence)
    
    # 清理内存
    del outputs
    del input_ids
    del attention_mask
    torch.cuda.empty_cache()
    
    return generated_texts, references, contexts, question_ids, answer_starts

# 故障注入推理函数（批次处理）
def faulty_inference_batch(dataset, tokenizer, corrupted_model, max_new_tokens=50):
    references, prompts, input_ids, attention_mask, question_ids, answer_starts, contexts, questions = get_batch_inputs(dataset, tokenizer, device)
    prompt_lens = [len(ids) for ids in input_ids]
    
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
    for i in range(batch_size):
        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        
        if "\n" in generated_text:
            generated_text = generated_text.split("\n")[0].strip()
        
        sentence_endings = re.split(r'(?<=[.!?])\s+', generated_text)
        first_sentence = sentence_endings[0].strip() if sentence_endings else generated_text.strip()
        generated_texts.append(first_sentence)
    
    del outputs
    del input_ids
    del attention_mask
    torch.cuda.empty_cache()
    
    return generated_texts, references, contexts, question_ids, answer_starts

# 实验设置
num_trials = 15000  # 总试验次数
results = []

# 获取批次信息
references, prompts, _, _, question_ids, answer_starts, contexts, questions = get_batch_inputs(selected_samples, tokenizer, device)

# 初始化故障注入器
dummy_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
seq_len = dummy_inputs["input_ids"].shape[1]

# 临时修改 forward 方法以提供 dummy decoder_input_ids
original_forward = model.forward
def dummy_forward(*args, **kwargs):
    # 处理位置参数
    if args:
        if len(args) > 0:
            kwargs['input_ids'] = args[0]
        if len(args) > 1:
            kwargs['decoder_input_ids'] = args[1]
    
    # 确保 input_ids 是 long 类型且在正确设备上
    if 'input_ids' in kwargs and kwargs['input_ids'] is not None:
        input_shape = kwargs['input_ids'].shape
        kwargs['input_ids'] = torch.randint(0, model.config.vocab_size, input_shape, dtype=torch.long, device=device)
    
    # 提供 dummy decoder_input_ids
    if 'decoder_input_ids' not in kwargs or kwargs['decoder_input_ids'] is None:
        batch_size = kwargs['input_ids'].shape[0]
        kwargs['decoder_input_ids'] = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    
    # 确保所有输入张量在同一设备上
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
    baseline_answers, baseline_references, baseline_contexts, baseline_question_ids, baseline_answer_starts = generate_batch(selected_samples, tokenizer, model)
except Exception as e:
    logging.error(f"基准推理失败: {str(e)}\n{traceback.format_exc()}")
    print(f"基准推理失败: {str(e)}")
    sys.exit(1)

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

# 故障注入试验（批次处理）
print(f"=== {num_trials} 次随机单比特注入试验（批次处理） ===")
i = 0
while i < num_trials:
    try:
        logging.info(f"开始试验 {i + 1}")
        pfi_model.reset_generate()  # 重置 generate 迭代记录
        pfi_model.reset_faults()  # 重置 fault 列表
        corrupted_model = nerr.random_neuron_single_bit_inj_batched(pfi_model)
        faulty_answers, faulty_references, faulty_contexts, faulty_question_ids, faulty_answer_starts = faulty_inference_batch(selected_samples, tokenizer, corrupted_model)
        fault_infos = pfi_model.last_faults
        
        print(f"[试验 {i+1}]")
        for idx in range(batch_size):
            print(f"样本 {idx+1} 答案: {faulty_answers[idx]}")
        print(f"   ↳ 故障信息: {fault_infos}")
        
        for fault_info in fault_infos:
            batch_idx = fault_info["batch"]
            sample_results[batch_idx]["faulty_outputs"].append({
                "trial": i + 1,
                "output": faulty_answers[batch_idx],
                "fault_info": fault_info
            })
        logging.info(f"试验 {i + 1} 成功完成。故障信息: {pfi_model.last_faults}")
        i += 1
    except Exception as e:
        logging.error(f"试验 {i + 1} 失败: {str(e)}\n{traceback.format_exc()}")
    finally:
        pfi_model.reset_fault_injection()
        if 'corrupted_model' in locals():
            del corrupted_model
        torch.cuda.empty_cache()

results = sample_results
print("\n")

# 保存结果
output_dir = "./results/t5-3b/squad"
with open(f"{output_dir}/t5-3b_fi_neuron_1bit_prefill_FP32.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"实验完成。结果保存至 '{output_dir}/t5-3b_fi_neuron_1bit_prefill_FP32.json'。")