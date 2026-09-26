import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# ================= 配置区域 =================
SCORE_JSON = "storage/gqs_score/amass.json"
CLASS_JSON = "storage/configs/amass_n20.json"

# 当前的权重设置 (请根据 score_multi_gpu_faster.py 保持一致)
WEIGHTS = {
    "foot_sliding": 1.0,
    "velocity_violation": 5,
    "self_collision": 0.01,
    "jerk": 0.01,
    "penetration": 10.0,
    "floating_frames_ratio": 200.0
}
# ===========================================

def load_data():
    print(f"Loading scores from {SCORE_JSON}...")
    with open(SCORE_JSON, 'r') as f:
        score_data = json.load(f)
    details = score_data.get("details", {})

    print(f"Loading classes from {CLASS_JSON}...")
    with open(CLASS_JSON, 'r') as f:
        class_data = json.load(f)

    # 建立 filename -> class_id 的映射
    file_to_class = {}
    for cid, paths in class_data.items():
        for p in paths:
            fname = os.path.basename(p)
            file_to_class[fname] = cid

    return details, file_to_class, class_data

def calculate_deductions(metrics, weights):
    """计算单项扣分"""
    deductions = {}
    total_deduction = 0.0
    for k, w in weights.items():
        val = metrics.get(k, 0.0)
        # 异常值处理：如果是 None 或 inf，设为 0 或跳过
        if val is None or not np.isfinite(val):
            val = 0.0
        score_loss = val * w
        deductions[k] = score_loss
        total_deduction += score_loss
    return deductions, total_deduction

def analyze_group(group_name, metrics_list):
    """分析一组数据的统计信息"""
    if not metrics_list:
        return None

    stats = {}
    n = len(metrics_list)

    # 将 list of dicts 转为 dict of lists 方便 numpy 计算
    # data_by_key: {'foot_sliding': [0.1, 0.2, ...], ...}
    data_by_key = defaultdict(list)
    total_deductions = []

    for m in metrics_list:
        deds, tot = calculate_deductions(m, WEIGHTS)
        total_deductions.append(tot)
        for k, v in deds.items():
            data_by_key[k].append(v)

    # 计算该组的总扣分均值，用于计算占比
    avg_total_loss = np.mean(total_deductions)
    if avg_total_loss < 1e-6: avg_total_loss = 1e-6 # 避免除零

    summary = []
    for k in WEIGHTS.keys():
        arr = np.array(data_by_key[k])

        # 1. 扣分率: 扣分 > 0.001 的比例
        penalty_rate = np.mean(arr > 0.001) * 100

        # 2. 扣分均值
        mean_deduction = np.mean(arr)

        # 3. 扣分最大值
        max_deduction = np.max(arr)

        # 4. 贡献占比 (该项平均扣分 / 总平均扣分)
        contribution = (mean_deduction / avg_total_loss) * 100

        summary.append({
            "Metric": k,
            "Penalty Rate (%)": f"{penalty_rate:.1f}%",
            "Mean Deduction": f"{mean_deduction:.2f}",
            "Max Deduction": f"{max_deduction:.2f}",
            "Contrib (%)": f"{contribution:.1f}%"
        })

    return pd.DataFrame(summary)

def main():
    if not os.path.exists(SCORE_JSON):
        print("Score file not found.")
        return

    details, file_to_class, class_data = load_data()

    # 1. 准备数据容器
    all_metrics = []
    class_metrics = defaultdict(list)

    valid_count = 0
    missing_class_count = 0

    for fname, mets in details.items():
        all_metrics.append(mets)

        if fname in file_to_class:
            cid = file_to_class[fname]
            class_metrics[cid].append(mets)
        else:
            missing_class_count += 1

    print(f"Total files in score json: {len(details)}")
    print(f"Files matched to classes: {len(details) - missing_class_count}")
    print("-" * 60)

    # 2. 全局分析
    print("\n=== [Overall Dataset Statistics] ===")
    df_all = analyze_group("Overall", all_metrics)
    print(df_all.to_string(index=False))

    # 3. 类别分析 (按 Class ID 排序)
    sorted_cids = sorted(class_metrics.keys(), key=lambda x: int(x) if x.isdigit() else x)

    # 为了防止输出太长，我们可以把结果汇总到一个大表格里，或者逐个打印
    # 这里选择：打印每个类的 "Top Contributor" (扣分最多的项) 和简要信息

    print("\n\n=== [Per-Class Breakdown] ===")
    print(f"{'Class ID':<10} | {'Files':<6} | {'Avg Score Loss':<15} | {'Main Penalty Source (Contrib %)'}")
    print("-" * 80)

    for cid in sorted_cids:
        mets = class_metrics[cid]
        deds_list = []
        for m in mets:
            _, t = calculate_deductions(m, WEIGHTS)
            deds_list.append(t)

        avg_loss = np.mean(deds_list)

        # 简单分析该类
        df = analyze_group(cid, mets)
        # 找到 Contribution 最大的项
        # 需要解析 string百分比回 float 排序，或者直接在 analyze_group 里返回数值
        # 这里简单处理：重新算一下 mean deduction 最大的
        sums = {}
        for k in WEIGHTS.keys():
            vals = [calculate_deductions(m, WEIGHTS)[0][k] for m in mets]
            sums[k] = np.mean(vals)

        main_source = max(sums, key=sums.get)
        main_contrib = (sums[main_source] / avg_loss * 100) if avg_loss > 1e-6 else 0.0

        print(f"{cid:<10} | {len(mets):<6} | {avg_loss:<15.2f} | {main_source} ({main_contrib:.1f}%)")

    # 如果需要查看某个特定类的详细表格，可以在这里修改代码指定打印
    # print("\nDetailed Stats for Class '0':")
    # print(analyze_group("0", class_metrics['0']).to_string(index=False))

if __name__ == "__main__":
    main()