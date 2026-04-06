import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import json
import logging
import traceback
import re
import sys
import random

# --- pfi imports ---
sys.path.append("/home/fanbao/vscodeWorkplace/pytorchfi")
import pytorchfi.neuron_error_models as nerr

# ---------------------- Config ----------------------
MODEL_LOCAL_PATH = "./models/qwen2-7b"
GSM8K_SELECTED_JSON = "./data/gsm8k/gsm8k_selected_samples.json"  # expected structure: {"samples": [{"question":..., "answer":...}, ...]}
RESULTS_DIR = "./results/qwen2-7b/gsm8k"
NUM_TRIALS = 15000
MAX_NEW_TOKENS = 128  # math can need a bit more room

# ---------------------- Logging ---------------------
logging.basicConfig(
    filename='fault_injection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ----------------- Load Model/Tokenizer -------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_LOCAL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
device = model.device

# ----------------- Load Selected Samples -------------
print("从保存的文件加载选中的样本 (GSM8K)...")
with open(GSM8K_SELECTED_JSON, "r") as f:
    gsm_data = json.load(f)

if isinstance(gsm_data, dict) and "samples" in gsm_data:
    gsm_samples = Dataset.from_list(gsm_data["samples"])
else:
    gsm_samples = Dataset.from_list(gsm_data)

batch_size = len(gsm_samples)
print(f"已加载 {batch_size} 个样本用于实验 (GSM8K)")

# ----------------- Helpers -----------------
# ========== Chain-of-Thought Prompt ==========
FEWSHOT_EXAMPLE = """Solve the following math problem step by step. Let's think step by step.
Example:
Q: John has 3 apples, he buys 2 more. How many does he have now?
A: John starts with 3. He buys 2. Total = 3 + 2 = 5.
Final Answer: 5
"""

def build_prompt(question):
    return (
        FEWSHOT_EXAMPLE
        + f"Now solve this problem:\nQ: {question}\nA:"
    )

def extract_final_answer(text):
    """与select_data_gsm8k.py一致，鲁棒提取最终答案"""
    if not isinstance(text, str):
        return ""
    m = re.search(r'\\?boxed\{([^}]*)\}', text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r'(final answer|answer|solution)[:\s]*([^\n\r]+)', text, flags=re.IGNORECASE)
    if m2:
        return m2.group(2).strip()
    m3 = re.search(r'[:=\-\u21D2]\s*([^\n\r]+)$', text.strip())
    if m3:
        return m3.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if re.search(r'\d', last):
            return last
        return last
    return text.strip()

def extract_number(s):
    if not isinstance(s, str):
        return ""
    matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', s)
    return matches[-1] if matches else ""

def get_batch_inputs(dataset, tokenizer, device):
    prompts, references, qids, contexts, questions = [], [], [], [], []
    for i in range(len(dataset)):
        q = dataset[i].get("question", "").strip()
        a = dataset[i].get("answer", "")
        ref = extract_final_answer(a)
        qid = dataset[i].get("id", str(i))

        prompt = build_prompt(q)
        prompts.append(prompt)
        references.append(ref)
        qids.append(qid)
        contexts.append("")
        questions.append(q)

    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attention_mask = enc["attention_mask"].to(device, non_blocking=True)
    return references, prompts, input_ids, attention_mask, qids, contexts, questions

def _decode_batch(outputs, prompt_lens, tokenizer, batch_sz):
    answers = []
    for i in range(batch_sz):
        gen = tokenizer.decode(outputs[i][prompt_lens[i]:], skip_special_tokens=True)
        # 直接用select的extract_final_answer
        ans = extract_final_answer(gen)
        answers.append(ans)
    return answers

def generate_batch(dataset, tokenizer, model, max_new_tokens=128):
    refs, prompts, input_ids, attention_mask, qids, ctxs, qs = get_batch_inputs(dataset, tokenizer, device)
    prompt_lens = attention_mask.sum(dim=1).cpu().tolist()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = _decode_batch(outputs.cpu().tolist(), prompt_lens, tokenizer, len(dataset))
    del outputs, input_ids, attention_mask
    torch.cuda.empty_cache()
    return decoded, refs, ctxs, qids, qs

def faulty_inference_batch(dataset, tokenizer, corrupted_model, max_new_tokens=128):
    refs, prompts, input_ids, attention_mask, qids, ctxs, qs = get_batch_inputs(dataset, tokenizer, device)
    prompt_lens = attention_mask.sum(dim=1).cpu().tolist()
    with torch.no_grad():
        outputs = corrupted_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = _decode_batch(outputs.cpu().tolist(), prompt_lens, tokenizer, len(dataset))
    del outputs, input_ids, attention_mask
    torch.cuda.empty_cache()
    return decoded, refs, ctxs, qids, qs


# ----------------- Experiment (GSM8K) -----------------
if __name__ == "__main__":
    refs, prompts, dummy_inp_ids, dummy_attn, qids, ctxs, qs = get_batch_inputs(gsm_samples, tokenizer, device)
    seq_len = dummy_inp_ids.shape[1]

    pfi_model = nerr.single_bit_flip_func(
        model,
        batch_size=batch_size,
        input_shape=[seq_len],
        layer_types=["all"],
        use_cuda=True,
    )
    print(pfi_model.print_pytorchfi_layer_summary())

    # baseline
    baseline_answers, baseline_refs, baseline_ctxs, baseline_qids, baseline_qs = generate_batch(
        gsm_samples, tokenizer, model
    )
    prompt_lens_baseline_answers = []
    print("=== 基准 (无故障注入) — GSM8K ===")
    for i in range(batch_size):
        print(f"样本 {i+1}:\n题目: {baseline_qs[i]}\n基准答案: {baseline_answers[i]}\n参考答案: {baseline_refs[i]}\n")
        baseline_answer_tokens = tokenizer(baseline_answers[i], return_tensors="pt", padding=True, truncation=True)
        input_ids_baseline_answers = baseline_answer_tokens["input_ids"].to(device, non_blocking=True)
        prompt_lens_baseline_answer = [len(ids) for ids in input_ids_baseline_answers]
        prompt_lens_baseline_answers.append(prompt_lens_baseline_answer[0])

    sample_results = []
    for i in range(batch_size):
        sample_results.append({
            "sample_id": i + 1,
            "question": baseline_qs[i],
            "ground_truth_answer": baseline_refs[i],
            "baseline_output": baseline_answers[i],
            "faulty_outputs": [],
        })

    print(f"=== {NUM_TRIALS} 次随机单比特注入试验（批次处理）— GSM8K ===")
    for t in range(NUM_TRIALS):
        try:
            logging.info(f"开始试验 {t + 1}")
            target_generate = []  
            for length in prompt_lens_baseline_answers:
                if length == 1:
                    target_generate.append(1)
                else:
                    # Generate a random integer between 2 and length (inclusive)
                    target_generate.append(random.randint(2, 128))
            pfi_model.target_generate = target_generate
            pfi_model.reset_generate()
            pfi_model.reset_faults()
            corrupted_model = nerr.random_neuron_two_bit_inj_batched(pfi_model)
            faulty_answers, _, _, _, _ = faulty_inference_batch(gsm_samples, tokenizer, corrupted_model)
            fault_infos = pfi_model.last_faults

            print(f"[试验 {t+1}]")
            for i in range(batch_size):
                print(f"样本 {i+1} 答案: {faulty_answers[i]}")
            print(f"   ↳ 故障信息: {fault_infos}")

            for fault in fault_infos:
                bidx = fault["batch"]
                sample_results[bidx]["faulty_outputs"].append({
                    "trial": t + 1,
                    "output": faulty_answers[bidx],
                    "fault_info": fault,
                })
            logging.info(f"试验 {t + 1} 成功完成。故障信息: {pfi_model.last_faults}")
        except Exception as e:
            # err = {
            #     "trial": t + 1,
            #     "error": str(e),
            #     "stack_trace": traceback.format_exc(),
            #     "fault_info": getattr(pfi_model, 'last_faults', None),
            # }
            # logging.error(f"试验 {t + 1} 失败: {err}")
            # for i in range(batch_size):
            #     sample_results[i]["faulty_outputs"].append({
            #         "trial": t + 1,
            #         "output": None,
            #         "fault_info": None,
            #         "error": err,
            #     })
            t -= 1  # 保证完成指定数量的成功试验
            continue

        # 移除hook并释放显存
        pfi_model.reset_fault_injection()
        del corrupted_model
        torch.cuda.empty_cache()
    outpath = f"{RESULTS_DIR}/qwen2-7b_fi_neuron_2bit_decode_FP32.json"
    with open(outpath, "w") as f:
        json.dump(sample_results, f, indent=4)
    print(f"实验完成。结果保存至 '{outpath}'。")
