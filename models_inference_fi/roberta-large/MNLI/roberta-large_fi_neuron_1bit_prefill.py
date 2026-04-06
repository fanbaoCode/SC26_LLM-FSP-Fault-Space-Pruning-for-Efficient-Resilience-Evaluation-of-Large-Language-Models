#!/usr/bin/env python3
# roberta_mnli_fi.py
import os
import sys
import json
import random,time
import logging
from tqdm import tqdm

import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./models/roberta-large"
SAMPLES_PATH = "./data/mnli/mnli_selected_samples.json"
NUM_TRIALS = 30000
OUTPUT_DIR = "./results/roberta-large/mnli"

# PyTorchFI path (adjust if needed)
sys.path.append("/home/fanbao/vscodeWorkplace/pytorchfi")
import pytorchfi.core as pfi_core
import pytorchfi.neuron_error_models as nerr

# logging
logging.basicConfig(
    filename='fault_injection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

def load_selected_samples(path):
    """
    Load selected samples. Supports:
      - a huggingface dataset saved with save_to_disk (directory)
      - a JSON file containing {"indices": [...], "samples": [...]}
      - a plain JSON list of samples
    Returns a HuggingFace Dataset.
    """
    if os.path.isdir(path):
        try:
            ds = datasets.load_from_disk(path)
            return ds
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset directory {path}: {e}")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        elif isinstance(data, list):
            samples = data
        else:
            raise ValueError("JSON file must be a list of samples or a dict with a 'samples' key.")
        return datasets.Dataset.from_list(samples)
    raise FileNotFoundError(f"Path not found: {path}")

# ========== Load samples ==========
print("加载 MNLI 数据集样本...")
ds = load_selected_samples(SAMPLES_PATH)
print(f"raw sample count: {len(ds)}")
# detect fields
cols = ds.column_names
if "premise" in cols and "hypothesis" in cols:
    sent1_key, sent2_key = "premise", "hypothesis"
elif "sentence1" in cols and "sentence2" in cols:
    sent1_key, sent2_key = "sentence1", "sentence2"
else:
    raise RuntimeError(f"无法识别样本字段，当前列名: {cols}")

if "label" not in cols:
    raise RuntimeError("样本中缺少 'label' 字段，无法继续。")

# ========== Load model & tokenizer ==========
print("加载模型与分词器...")
# if local folder not found, AutoTokenizer.from_pretrained will try hub id
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# load model (float16 if GPU available)
map_kwargs = {}
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model = model.to("cuda:0")
model.eval()
device = model.device
print(f"模型加载完成，device={device}")

# ========== Helper: prepare batch tensors ==========
def get_batch_inputs(dataset, tokenizer, device, max_length=128):
    input_ids_list = []
    attention_masks = []
    labels = []
    sents1 = []
    sents2 = []

    for i in range(len(dataset)):
        s1 = dataset[i][sent1_key]
        s2 = dataset[i][sent2_key]
        lbl = int(dataset[i]["label"])

        enc = tokenizer(
            s1, s2,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids_list.append(enc["input_ids"].squeeze(0))
        attention_masks.append(enc["attention_mask"].squeeze(0))

        labels.append(lbl)
        sents1.append(s1)
        sents2.append(s2)

    input_ids = torch.stack(input_ids_list).to(device)
    attention_mask = torch.stack(attention_masks).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)

    return labels, input_ids, attention_mask, sents1, sents2

def generate_batch(dataset, tokenizer, model, max_length=128):
    labels, input_ids, attention_mask, sents1, sents2 = get_batch_inputs(dataset, tokenizer, device, max_length)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
    del outputs, input_ids, attention_mask
    torch.cuda.empty_cache()
    return preds.cpu().tolist(), labels.cpu().tolist(), sents1, sents2

def faulty_inference_batch(dataset, tokenizer, corrupted_model, max_length=128):
    labels, input_ids, attention_mask, sents1, sents2 = get_batch_inputs(dataset, tokenizer, device, max_length)
    with torch.no_grad():
        outputs = corrupted_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
    del outputs, input_ids, attention_mask
    torch.cuda.empty_cache()
    return preds.cpu().tolist(), labels.cpu().tolist(), sents1, sents2

# ========== Init PyTorchFI injector ==========
print("初始化故障注入器 (PyTorchFI)...")
dummy_inputs = tokenizer(
    [ds[i][sent1_key] for i in range(len(ds))],
    [ds[i][sent2_key] for i in range(len(ds))],
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
seq_len = dummy_inputs["input_ids"].shape[1]

pfi_model = nerr.single_bit_flip_func(
    model,
    batch_size=len(ds),
    input_shape=[seq_len],
    layer_types=["all"],
    use_cuda=torch.cuda.is_available()
)
print(pfi_model.print_pytorchfi_layer_summary())

# ========== Baseline inference ==========
baseline_preds, baseline_labels, sents1, sents2 = generate_batch(ds, tokenizer, model)
print("=== 基准 (无故障注入) ===")
for i in range(len(baseline_preds)):
    print(f"样本 {i+1}:")
    print(f"  {sent1_key}: {sents1[i]}")
    print(f"  {sent2_key}: {sents2[i]}")
    print(f"  label: {baseline_labels[i]} ({label_map.get(baseline_labels[i], 'UNK')})  pred: {baseline_preds[i]} ({label_map.get(baseline_preds[i], 'UNK')})")
    print()

sample_results = []
for idx in range(len(ds)):
    sample_results.append({
        "sample_id": idx + 1,
        sent1_key: sents1[idx],
        sent2_key: sents2[idx],
        "ground_truth_label": int(baseline_labels[idx]),
        "baseline_output": int(baseline_preds[idx]),
        "baseline_output_name": label_map.get(baseline_preds[idx], str(baseline_preds[idx])),
        "faulty_outputs": []
    })

# ========== Fault injection trials ==========
print(f"=== {NUM_TRIALS} 次随机单比特注入试验（批次处理） ===")
for t in range(NUM_TRIALS):
    # try:
    logging.info(f"开始试验 {t+1}")
    # resets
    pfi_model.reset_generate()
    pfi_model.reset_faults()
    corrupted_model = nerr.random_neuron_single_bit_inj_batched(pfi_model)
    faulty_preds, faulty_labels, faulty_sents1, faulty_sents2 = faulty_inference_batch(ds, tokenizer, corrupted_model)
    fault_infos = getattr(pfi_model, "last_faults", None)

    print(f"[试验 {t+1}]")
    for i in range(len(faulty_preds)):
        print(f"  样本 {i+1} 预测: {faulty_preds[i]} ({label_map.get(faulty_preds[i], 'UNK')})")
    print(f"   ↳ 故障信息: {fault_infos}")

    # record per-fault entries (if any)
    if fault_infos:
        for finfo in fault_infos:
            batch_idx = finfo.get("batch", 0)
            sample_results[batch_idx]["faulty_outputs"].append({
                "trial": t+1,
                "output": int(faulty_preds[batch_idx]),
                "output_name": label_map.get(int(faulty_preds[batch_idx]), str(faulty_preds[batch_idx])),
                "fault_info": finfo
            })

    logging.info(f"试验 {t+1} 完成。故障信息: {fault_infos}")
    # except Exception as e:
    #     logging.exception(f"试验 {t+1} 出错: {e}")
    #     NUM_TRIALS += 1  # ensure we do NUM_TRIALS
    #     continue
    pfi_model.reset_fault_injection()
    del corrupted_model
    torch.cuda.empty_cache()

# ========== Save results ==========
out_path = os.path.join(OUTPUT_DIR, "roberta-large_fi_neuron_1bit_FP32.json")
with open(out_path, "w", encoding="utf-8") as fout:
    json.dump(sample_results, fout, indent=4, ensure_ascii=False)

print(f"实验完成。结果保存至 '{out_path}'.")
