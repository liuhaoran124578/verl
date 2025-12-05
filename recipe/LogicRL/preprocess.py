#!/usr/bin/env python3
"""
preprocess.py
将原始 parquet 转换为符合 Verl RLHFDataset 的格式
"""
import argparse
import os
import re
from typing import Any, Dict, List

import numpy as np
from datasets import load_dataset


def convert_numpy_types(data: Any) -> Any:
    """Convert numpy scalars/arrays to Python native types recursively."""
    if isinstance(data, np.ndarray):
        return [convert_numpy_types(d) for d in data.tolist()]
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        return float(data)
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_numpy_types(v) for v in data]
    return data


def clean_text_remove_assistant_segments(text: str) -> str:
    """Remove embedded assistant segments from text."""
    if not isinstance(text, str):
        return text
    markers = ["<|im_start|>assistant", "<|im_start|>assistant\n", "<|im_start|>assistant "]
    cleaned = text
    for m in markers:
        if m in cleaned:
            cleaned = cleaned.split(m, 1)[0]
    cleaned = re.sub(r"</?think>", "", cleaned)
    cleaned = re.sub(r"</?answer>", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def normalize_newlines(text: str) -> str:
    """
    确保文本中的换行符是真实的 \n，而不是字面量 \\n
    """
    if not isinstance(text, str):
        return text
    # 如果只包含字面量 \n（没有真换行符），转换之
    if '\\n' in text and '\n' not in text:
        text = text.replace('\\n', '\n')
    return text


def build_assistant_reference(cot_head: str,
                              cot_steps: List[str],
                              cot_foot: str,
                              answer: str) -> str:
    """
    构建 assistant 参考内容：<think>...</think><answer>...</answer>
    """
    cot_steps = [str(s).strip() for s in (cot_steps or []) if s and str(s).strip()]
    reasoning_parts = []
    if cot_head:
        reasoning_parts.append(cot_head.strip())
    if cot_steps:
        reasoning_parts.append(" ".join(cot_steps))
    if cot_foot:
        reasoning_parts.append(cot_foot.strip())
    reasoning_process = " ".join(reasoning_parts).strip()
    
    # ✅ 规范化答案中的换行符
    answer = answer.strip() if answer else ''
    answer = normalize_newlines(answer)
    
    think_block = f"<think>{reasoning_process}</think>"
    answer_block = f"<answer>{answer}</answer>"
    return f"{think_block}{answer_block}"


def validate_response_structure(response_content: str) -> bool:
    """Check exactly one <think>...</think> and one <answer>...</answer> and proper order."""
    if not isinstance(response_content, str):
        return False
    think_open = len(re.findall(r"<think>", response_content))
    think_close = len(re.findall(r"</think>", response_content))
    answer_open = len(re.findall(r"<answer>", response_content))
    answer_close = len(re.findall(r"</answer>", response_content))
    
    if not (think_open == think_close == answer_open == answer_close == 1):
        return False
    
    first_think_open = response_content.find("<think>")
    first_think_close = response_content.find("</think>")
    first_answer_open = response_content.find("<answer>")
    first_answer_close = response_content.find("</answer>")
    
    if not (0 <= first_think_open < first_think_close < first_answer_open < first_answer_close):
        return False
    return True


def validate_answer_format(answer_content: str, expected_count: int = 3) -> bool:
    """
    验证 answer 内容是否严格符合格式：
    (1) Name is a knight/knave
    (2) Name is a knight/knave
    ...
    """
    lines = answer_content.strip().split('\n')
    if len(lines) != expected_count:
        return False
    
    for i, line in enumerate(lines, 1):
        # 严格匹配格式
        if not re.match(rf'^\({i}\)\s+[A-Za-z]+\s+is\s+a\s+(knight|knave)$', line.strip(), re.IGNORECASE):
            return False
    return True


def make_map_fn(split: str):
    """
    返回一个映射函数，用于 datasets.Dataset.map
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        
        system_msg = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight, (2) ... </answer>."""

        # 1) 获取用户问题
        user_msg = example.get("quiz", "")
        if not user_msg:
            orig_prompt = example.get("prompt", None)
            if orig_prompt:
                try:
                    if isinstance(orig_prompt, (list, tuple)):
                        first = orig_prompt[0]
                        if isinstance(first, dict) and "content" in first:
                            user_msg = first["content"]
                        else:
                            user_msg = str(first)
                    elif isinstance(orig_prompt, dict) and "content" in orig_prompt:
                        user_msg = orig_prompt["content"]
                    else:
                        user_msg = str(orig_prompt)
                except Exception:
                    user_msg = str(orig_prompt)

        user_msg = clean_text_remove_assistant_segments(user_msg)

        # 2) 构建 assistant 参考答案
        cot_head = (example.get("cot_head") or "") or ""
        cot_steps = convert_numpy_types(example.get("cot_repeat_steps", [])) or []
        cot_foot = (example.get("cot_foot") or "") or ""
        
        if isinstance(cot_steps, str):
            cot_steps = [cot_steps]
        
        # ✅ 使用 solution_text_format 作为标准答案
        solution_text_format = (example.get("solution_text_format") or "").strip()
        solution_text_format = normalize_newlines(solution_text_format)
        
        assistant_content = build_assistant_reference(
            cot_head, cot_steps, cot_foot, solution_text_format
        )

        # 3) reward_model 结构
        statements = example.get("statements", "") or ""
        statements = str(statements) if statements else ""
        
        reward_model = {
            "style": "rule",
            "ground_truth": {
                "solution_text_format": solution_text_format,
                "statements": statements
            }
        }

        # 4) 其他字段
        names = convert_numpy_types(example.get("names", []))
        solution = convert_numpy_types(example.get("solution", []))

        out = {
            "data_source": example.get("data_source", "kk_logic"),
            "prompt": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "response": {"role": "assistant", "content": assistant_content},
            "names": names,
            "solution": solution,
            "solution_text": example.get("solution_text", "") or "",
            "ability": example.get("ability", "logic"),
            "reward_model": reward_model,
            "extra_info": {"split": split, "index": int(idx)}
        }
        return out
    return process_fn


def process_dataset(raw_parquet_path: str, output_parquet_path: str, split: str, preview: int = 5):
    """
    加载、预处理、验证、保存数据集
    """
    assert os.path.exists(raw_parquet_path), f"Input file not found: {raw_parquet_path}"
    os.makedirs(os.path.dirname(output_parquet_path) or ".", exist_ok=True)

    print(f"[INFO] Loading raw dataset from: {raw_parquet_path}")
    ds = load_dataset("parquet", data_files={split: raw_parquet_path})[split]

    print(f"[INFO] Number of raw examples: {len(ds)}")
    mapper = make_map_fn(split)
    print("[INFO] Applying preprocessing...")
    processed = ds.map(mapper, with_indices=True, desc=f"Processing {split}")

    # 验证前 N 个样本
    n_check = min(preview, len(processed))
    print(f"\n[INFO] Validating first {n_check} examples...")
    print("="*80)
    
    issues = 0
    for i in range(n_check):
        ex = processed[i]
        resp = ex.get("response", {})
        content = resp.get("content", "") if isinstance(resp, dict) else ""
        
        # 结构验证
        valid_structure = validate_response_structure(content)
        
        # 格式验证
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        valid_format = False
        if answer_match:
            answer_content = answer_match.group(1).strip()
            names = ex.get("names", [])
            expected_count = len(names) if names else 3
            valid_format = validate_answer_format(answer_content, expected_count)
            
            # 检查换行符
            has_real_newline = '\n' in answer_content
            has_literal = '\\n' in answer_content
            
            print(f"\n[{i}] Structure: {'✅' if valid_structure else '❌'}  |  Format: {'✅' if valid_format else '❌'}")
            print(f"     Real newlines: {has_real_newline}  |  Literal \\n: {has_literal}")
            
            if not valid_format:
                issues += 1
                print(f"     ⚠️  Answer content:")
                print(f"     {answer_content[:200]}")
        else:
            print(f"\n[{i}] Structure: ❌  |  No <answer> tag found")
            issues += 1

    print("\n" + "="*80)
    if issues > 0:
        print(f"⚠️  Found {issues} issues in first {n_check} examples.")
        print("    Please review the output above and check your source data.")
    else:
        print(f"✅ All {n_check} examples validated successfully!")

    print(f"\n[INFO] Saving to: {output_parquet_path}")
    processed.to_parquet(output_parquet_path)
    print("[INFO] Done.\n")


def parse_args():
    p = argparse.ArgumentParser(description="Convert raw parquet to RLHF-format for Verl")
    p.add_argument("--input", "-i", required=True, help="Raw parquet input file")
    p.add_argument("--output", "-o", required=True, help="Output processed parquet path")
    p.add_argument("--split", default="train", choices=["train", "test", "validation"],
                   help="Split name")
    p.add_argument("--preview", type=int, default=5, help="Number of examples to validate")
    return p.parse_args()


def main():
    args = parse_args()
    process_dataset(args.input, args.output, args.split, preview=args.preview)


if __name__ == "__main__":
    main()