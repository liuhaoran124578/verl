#!/usr/bin/env bash
set -e  # 遇到错误立即停止
set -u  # 使用未定义的变量报错
set -o pipefail

# ================= 1. 路径与环境配置 (仿照 VERL 风格) =================

# 获取当前脚本所在的目录 (即 recipe/LogicRL)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 定义主目录，默认为当前脚本目录
export LOGICRL_HOME=${LOGICRL_HOME:-"${SCRIPT_DIR}"}

# 定义原始数据存放路径 (Source)
# 结构: recipe/LogicRL/data/kk/instruct
export RAW_DATA_DIR=${RAW_DATA_DIR:-"${LOGICRL_HOME}/data/kk/instruct"}

# 定义处理后数据存放路径 (Target)
# 结构: recipe/LogicRL/data/processed_data
export PROCESSED_DATA_DIR=${PROCESSED_DATA_DIR:-"${LOGICRL_HOME}/data/processed_data"}

# 定义预处理脚本路径
export PREPROCESS_SCRIPT="${LOGICRL_HOME}/preprocess.py"

# GitHub 下载基地址 (注意：必须使用 raw.githubusercontent 才能下载文件)
GITHUB_BASE_URL="https://raw.githubusercontent.com/Unakar/Logic-RL/main/data/kk/instruct"

echo "========================================================="
echo "   Working Directory: ${LOGICRL_HOME}"
echo "   Raw Data Dir:      ${RAW_DATA_DIR}"
echo "   Processed Dir:     ${PROCESSED_DATA_DIR}"
echo "========================================================="

# 确保 Python 依赖存在
pip install pandas pyarrow fastparquet datasets requests > /dev/null 2>&1 || true

# ================= 2. 自动下载数据 (Download) =================
echo ">>> [Step 1] Downloading data from GitHub..."

mkdir -p "${RAW_DATA_DIR}"

# 定义需要下载的文件夹列表
FOLDERS=("3ppl" "4ppl" "5ppl" "6ppl" "7ppl")

for FOLDER in "${FOLDERS[@]}"; do
    TARGET_DIR="${RAW_DATA_DIR}/${FOLDER}"
    mkdir -p "${TARGET_DIR}"
    
    for FILE in "train.parquet" "test.parquet"; do
        LOCAL_FILE="${TARGET_DIR}/${FILE}"
        if [ ! -f "${LOCAL_FILE}" ]; then
            REMOTE_URL="${GITHUB_BASE_URL}/${FOLDER}/${FILE}"
            echo "Downloading: ${FOLDER}/${FILE} ..."
            wget -q -O "${LOCAL_FILE}" "${REMOTE_URL}" || echo "  [Warning] Failed to download ${REMOTE_URL}"
        else
            echo "Skipping: ${FOLDER}/${FILE} (Already exists)"
        fi
    done
done

# ================= 3. 合并数据 3-4-5 (Merge) =================
echo ">>> [Step 2] Merging 3ppl, 4ppl, 5ppl..."

# 为了不增加额外的 .py 文件，这里使用 python -c 直接执行合并逻辑
# 这段 Python 代码会读取 RAW_DATA_DIR 下的 3,4,5ppl 并生成 mix_3_4_5ppl
python3 -c "
import pandas as pd
import os

base_dir = os.environ['RAW_DATA_DIR']
folders = ['3ppl', '4ppl', '5ppl']
output_folder = 'mix_3_4_5ppl'
os.makedirs(os.path.join(base_dir, output_folder), exist_ok=True)

for split in ['train', 'test']:
    dfs = []
    for f in folders:
        path = os.path.join(base_dir, f, f'{split}.parquet')
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))
    
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(base_dir, output_folder, f'{split}.parquet')
        merged.to_parquet(out_path)
        print(f'  Merged {split}: {len(merged)} rows -> {out_path}')
    else:
        print(f'  No data found for split {split}')
"

# ================= 4. 数据预处理 (Preprocess) =================
echo ">>> [Step 3] Preprocessing all datasets..."

# 所有需要处理的数据集 (包含刚刚生成的 mix)
TASKS=("3ppl" "4ppl" "5ppl" "6ppl" "7ppl" "mix_3_4_5ppl")

for TASK in "${TASKS[@]}"; do
    echo "Processing Task: ${TASK}"
    
    INPUT_PATH="${RAW_DATA_DIR}/${TASK}"
    OUTPUT_PATH="${PROCESSED_DATA_DIR}/${TASK}"
    
    # 确保输出目录存在
    mkdir -p "${OUTPUT_PATH}"

    # 处理 Train
    if [ -f "${INPUT_PATH}/train.parquet" ]; then
        python "${PREPROCESS_SCRIPT}" \
            --input "${INPUT_PATH}/train.parquet" \
            --output "${OUTPUT_PATH}/train.parquet" \
            --split train \
            --preview 5
    fi

    # 处理 Test
    if [ -f "${INPUT_PATH}/test.parquet" ]; then
        python "${PREPROCESS_SCRIPT}" \
            --input "${INPUT_PATH}/test.parquet" \
            --output "${OUTPUT_PATH}/test.parquet" \
            --split test \
            --preview 5
    fi
done

echo "========================================================="
echo "✅ All Done! Processed data available at:"
echo "${PROCESSED_DATA_DIR}"
echo "========================================================="