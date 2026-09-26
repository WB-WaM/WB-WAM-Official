"""Build the portable HGPT replay downsampling comparison report."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

SEGMENT_LABELS = {
    "straight_walk": "直线行走",
    "stationary_action": "原地动作",
    "full_body_turn": "全身转向",
    "mild_waist_bend": "轻微弯腰",
}
METHOD_ORDER = {
    "causal_latest_20hz": 0,
    "interpolate_20hz": 1,
    "master_50hz": 2,
}


def _by_method(rows: list[dict], segment: str) -> dict[str, dict]:
    return {row["method"]: row for row in rows if row.get("segment") == segment}


def _round_rows(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        for field in fields:
            if field in item:
                item[field] = round(float(item[field]), 3)
        output.append(item)
    return output


def _select_rows(table: str, rows: list[dict], sql: str) -> list[dict]:
    """Materialize reviewed rows and execute the SQL saved in source metadata."""
    if not rows:
        raise ValueError(f"cannot query empty dataset {table}")
    fields = list(rows[0])
    for row in rows:
        if list(row) != fields:
            raise ValueError(f"inconsistent columns in dataset {table}")
    column_types = []
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        is_numeric = values and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        )
        column_types.append("REAL" if is_numeric else "TEXT")
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        columns = ", ".join(
            f'"{field}" {column_type}'
            for field, column_type in zip(fields, column_types, strict=True)
        )
        connection.execute(f'CREATE TABLE "{table}" ({columns})')
        field_list = ", ".join(f'"{field}"' for field in fields)
        placeholders = ", ".join("?" for _ in fields)
        connection.executemany(
            f'INSERT INTO "{table}" ({field_list}) VALUES ({placeholders})',
            ([row[field] for field in fields] for row in rows),
        )
        return [dict(row) for row in connection.execute(sql).fetchall()]
    finally:
        connection.close()


def _sql_source(
    source_id: str,
    label: str,
    table: str,
    sql: str,
    generated_at: str,
    metric_definitions: list[str],
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": (
            "tracker/humanoid_gpt/reports/downsampling_replay_analysis/analysis.json"
        ),
        "query": {
            "engine": "SQLite 3 in-memory normalized analysis output",
            "sql": sql,
            "language": "sql",
            "description": (
                "The report builder materializes the reviewed analysis rows in an "
                "in-memory SQLite table and executes this exact query."
            ),
            "executed_at": generated_at,
            "tables_used": [table],
            "filters": [
                "Common 1,421-frame support at 50 Hz (28.4 s)",
                "First 0.5 s loader transition excluded from segment metrics",
                "Production SE(2) first-frame alignment and EMA enabled",
            ],
            "metric_definitions": metric_definitions,
        },
    }


def build_artifact(analysis_path: Path, output_path: Path) -> None:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    reconstruction = []
    for row in analysis["reconstruction"]:
        if row["method"] == "master_50hz":
            continue
        reconstruction.append(
            {
                **row,
                "segment_label": SEGMENT_LABELS[row["segment"]],
            }
        )
    reconstruction = _round_rows(
        reconstruction,
        (
            "root_xy_mae_mm",
            "root_yaw_mae_deg",
            "joint_mae_deg",
            "kpt_position_mae_mm",
            "kpt_rotation_mae_deg",
        ),
    )

    tracking = [
        {**row, "segment_label": SEGMENT_LABELS[row["segment"]]}
        for row in analysis["tracking"]
    ]
    tracking = _round_rows(
        tracking,
        (
            "e2e_root_xy_mae_mm",
            "e2e_root_yaw_mae_deg",
            "e2e_joint_mae_deg",
            "own_root_xy_mae_mm",
            "own_root_yaw_mae_deg",
            "own_joint_mae_deg",
            "own_kpt_position_mae_mm",
            "own_kpt_rotation_mae_deg",
        ),
    )

    geometry = sorted(
        analysis["straight_geometry"], key=lambda row: METHOD_ORDER[row["method"]]
    )
    special = sorted(
        analysis["turn_and_bend"], key=lambda row: METHOD_ORDER[row["method"]]
    )
    locomotion = []
    for geometry_row, special_row in zip(geometry, special, strict=True):
        locomotion.append(
            {
                "method": geometry_row["method_label"],
                "straight_completion_pct": round(
                    100.0 * geometry_row["distance_completion_ratio"], 2
                ),
                "straight_cross_track_rms_mm": round(
                    geometry_row["cross_track_rms_mm"], 2
                ),
                "straight_heading_error_deg": round(
                    geometry_row["heading_error_deg"], 3
                ),
                "turn_actual_excursion_deg": round(
                    special_row["turn_actual_excursion_deg"], 2
                ),
                "turn_yaw_mae_deg": round(special_row["turn_yaw_mae_deg"], 3),
            }
        )
    bend = [
        {
            "method": row["method_label"],
            "reference_peak_deg": round(row["bend_reference_peak_deg"], 3),
            "actual_peak_deg": round(row["bend_actual_peak_deg"], 3),
            "waist_pitch_mae_deg": round(row["bend_waist_pitch_mae_deg"], 3),
        }
        for row in special
    ]

    straight_tracking = _by_method(analysis["tracking"], "straight_walk")
    action_tracking = _by_method(analysis["tracking"], "stationary_action")
    turn_tracking = _by_method(analysis["tracking"], "full_body_turn")
    reconstruction_by_segment = {
        segment: _by_method(analysis["reconstruction"], segment)
        for segment in SEGMENT_LABELS
    }

    best_20_straight_joint = min(
        straight_tracking[method]["e2e_joint_mae_deg"]
        for method in ("causal_latest_20hz", "interpolate_20hz")
    )
    best_20_action_joint = min(
        action_tracking[method]["e2e_joint_mae_deg"]
        for method in ("causal_latest_20hz", "interpolate_20hz")
    )
    best_20_turn_yaw = min(
        turn_tracking[method]["e2e_root_yaw_mae_deg"]
        for method in ("causal_latest_20hz", "interpolate_20hz")
    )
    direct_straight_joint = straight_tracking["master_50hz"]["e2e_joint_mae_deg"]
    direct_action_joint = action_tracking["master_50hz"]["e2e_joint_mae_deg"]
    direct_turn_yaw = turn_tracking["master_50hz"]["e2e_root_yaw_mae_deg"]
    straight_reduction = (
        100.0
        * (best_20_straight_joint - direct_straight_joint)
        / best_20_straight_joint
    )
    action_reduction = (
        100.0 * (best_20_action_joint - direct_action_joint) / best_20_action_joint
    )
    turn_reduction = 100.0 * (best_20_turn_yaw - direct_turn_yaw) / best_20_turn_yaw
    reconstruction_gains = []
    for rows in reconstruction_by_segment.values():
        causal = rows["causal_latest_20hz"]["joint_mae_deg"]
        interpolated = rows["interpolate_20hz"]["joint_mae_deg"]
        reconstruction_gains.append(100.0 * (causal - interpolated) / causal)

    analysis_source = {
        "id": "analysis",
        "label": "HGPT replay downsampling analysis",
        "path": (
            "tracker/humanoid_gpt/reports/downsampling_replay_analysis/analysis.json"
        ),
        "query": {
            "engine": "Python 3.12 + MuJoCo",
            "query": (
                "G1_VERSION=5010 conda run --no-capture-output -n h-gpt "
                "python tracker/humanoid_gpt/scripts/"
                "analyze_replay_downsampling.py"
            ),
            "language": "shell",
            "description": (
                "Loads all three references through the production offline loader, "
                "runs the HGPT tracking policy in deterministic headless MuJoCo, "
                "and calculates segment-level reconstruction and tracking errors."
            ),
            "executed_at": generated_at,
            "tables_used": [
                "datasets/humanoid_gpt/hgpt_master_50hz/replay_20hz/"
                "mode2_causal_latest_20hz.npz",
                "datasets/humanoid_gpt/hgpt_master_50hz/replay_20hz/"
                "mode3_interpolate_20hz.npz",
                "datasets/humanoid_gpt/hgpt_master_50hz/replay_20hz/"
                "mode4_master_50hz.npz",
            ],
            "filters": [
                "Common 1,421-frame support at 50 Hz (28.4 s)",
                "First 0.5 s loader transition excluded from segment metrics",
                "All references use production SE(2) first-frame alignment and EMA",
            ],
            "metric_definitions": [
                "Reference reconstruction error compares a processed 20 Hz reference "
                "with the processed direct-50 Hz reference at the same 50 Hz timestamps.",
                "End-to-end joint MAE is the mean absolute 29-joint position error "
                "between simulated robot qpos and the direct-50 Hz reference.",
                "Straight completion is simulated XY displacement divided by the "
                "direct-50 Hz reference displacement over the straight segment.",
                "Turn yaw MAE is wrapped absolute pelvis yaw error against the "
                "direct-50 Hz reference over the full-body turn segment.",
            ],
        },
    }

    reconstruction_sql = """
SELECT *
FROM reconstruction
ORDER BY CASE segment
    WHEN 'straight_walk' THEN 0
    WHEN 'stationary_action' THEN 1
    WHEN 'full_body_turn' THEN 2
    ELSE 3 END,
  CASE method
    WHEN 'causal_latest_20hz' THEN 0
    ELSE 1 END
""".strip()
    tracking_sql = """
SELECT *
FROM tracking
ORDER BY CASE segment
    WHEN 'straight_walk' THEN 0
    WHEN 'stationary_action' THEN 1
    WHEN 'full_body_turn' THEN 2
    ELSE 3 END,
  CASE method
    WHEN 'causal_latest_20hz' THEN 0
    WHEN 'interpolate_20hz' THEN 1
    ELSE 2 END
""".strip()
    locomotion_sql = """
SELECT *
FROM locomotion
ORDER BY straight_completion_pct DESC
""".strip()
    bend_sql = """
SELECT *
FROM bend
ORDER BY waist_pitch_mae_deg ASC
""".strip()
    reconstruction = _select_rows("reconstruction", reconstruction, reconstruction_sql)
    tracking = _select_rows("tracking", tracking, tracking_sql)
    locomotion = _select_rows("locomotion", locomotion, locomotion_sql)
    bend = _select_rows("bend", bend, bend_sql)

    reconstruction_source = _sql_source(
        "reconstruction_source",
        "Processed reference reconstruction metrics",
        "reconstruction",
        reconstruction_sql,
        generated_at,
        [
            "Joint MAE is mean absolute 29-joint position error between a processed "
            "20 Hz reference and the processed direct-50 Hz reference."
        ],
    )
    tracking_source = _sql_source(
        "tracking_source",
        "HGPT MuJoCo end-to-end tracking metrics",
        "tracking",
        tracking_sql,
        generated_at,
        [
            "End-to-end joint MAE is mean absolute 29-joint position error between "
            "simulated qpos and the direct-50 Hz reference."
        ],
    )
    locomotion_source = _sql_source(
        "locomotion_source",
        "Straight walking and full-body turn metrics",
        "locomotion",
        locomotion_sql,
        generated_at,
        [
            "Straight completion is simulated XY displacement divided by the "
            "direct-50 Hz reference displacement.",
            "Turn yaw MAE is wrapped absolute pelvis yaw error against the "
            "direct-50 Hz reference.",
        ],
    )
    bend_source = _sql_source(
        "bend_source",
        "Mild waist-bend tracking metrics",
        "bend",
        bend_sql,
        generated_at,
        [
            "Waist pitch MAE is absolute motor-14 waist pitch error against the "
            "direct-50 Hz reference over the mild-bend proxy frames."
        ],
    )
    sources = [
        analysis_source,
        reconstruction_source,
        tracking_source,
        locomotion_source,
        bend_source,
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# HGPT 20 Hz 下采样与 50 Hz Replay 定量对比",
            "layout": "full",
        },
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "analysis",
            "layout": "full",
            "body": (
                "## 技术结论\n\n"
                "**直接 50 Hz replay 是这条轨迹上的首选。** 相对表现最好的 "
                f"20 Hz 方案，它把直走段端到端关节 MAE 降低 {straight_reduction:.1f}%，"
                f"原地动作段降低 {action_reduction:.1f}%，并把全身转向 yaw MAE 从 "
                f"{best_20_turn_yaw:.2f}° 降到 {direct_turn_yaw:.2f}°（降低 "
                f"{turn_reduction:.1f}%）。\n\n"
                "**如果必须保存 20 Hz，离线数据优先使用 interpolation。** 它在送入 "
                f"policy 前的四类片段中，将关节重建 MAE 比 causal-latest 稳定降低 "
                f"{min(reconstruction_gains):.1f}%–{max(reconstruction_gains):.1f}%。"
                "不过进入 HGPT+MuJoCo 后，两种 20 Hz 方法差距很小且方向不一致，"
                "因此不要把 interpolation 解释为控制效果必然更好。\n\n"
                "**causal-latest 只在实时、不能使用未来帧时有明确优势。** 对离线采集后"
                "再 replay 的场景，它没有比 interpolation 更高的保真度。"
            ),
        },
        {
            "id": "tracking_heading",
            "type": "markdown",
            "sourceId": "analysis",
            "layout": "full",
            "body": (
                "## 直接 50 Hz 对原始动作的端到端跟踪最稳定\n\n"
                "下图以原始 50 Hz 参考为共同基准，比较 MuJoCo 中机器人实际 29 关节"
                "位置的平均绝对误差。直接 50 Hz 在所有片段都最低；20 Hz 的平滑虽然"
                "有时让机器人更容易跟随自身参考，但会偏离原始动作，因此这里不采用"
                "“对自身平滑参考更小”的误差作为优胜标准。"
            ),
        },
        {
            "id": "tracking_chart_block",
            "type": "chart",
            "chartId": "tracking_joint_chart",
            "layout": "full",
        },
        {
            "id": "reconstruction_heading",
            "type": "markdown",
            "sourceId": "analysis",
            "layout": "full",
            "body": (
                "## interpolation 更忠实地重建 20 Hz 参考，但优势约为 3%–4%\n\n"
                "该图只看输入 policy 之前的参考轨迹误差。四个片段中 interpolation "
                "均低于 causal-latest，方向一致但幅度不大。两种 20 Hz 都明显偏离"
                "直接 50 Hz，主要原因不是插值本身，而是 production loader 在 20 Hz "
                "数据上先应用相同的 EMA α=0.8：其等效时间常数约 224 ms，50 Hz "
                "只有约 90 ms，20 Hz 参考额外产生约 120 ms 的低频延迟。"
            ),
        },
        {
            "id": "reconstruction_chart_block",
            "type": "chart",
            "chartId": "reconstruction_joint_chart",
            "layout": "full",
        },
        {
            "id": "locomotion_heading",
            "type": "markdown",
            "sourceId": "analysis",
            "layout": "full",
            "body": (
                "## 直走与转向的主要损失来自 20 Hz 管线的相位滞后\n\n"
                "直走参考位移为 3.42 m。直接 50 Hz 完成 68.9%，两种 20 Hz 只完成 "
                "58% 左右；三者横向 RMS 漂移都约 85–90 mm，说明“走不够远”比"
                "“方向偏斜”更突出。转向目标 excursion 为 90.82°，直接 50 Hz 实现 "
                "81.91°，20 Hz 仅约 77°。"
            ),
        },
        {
            "id": "locomotion_table_block",
            "type": "table",
            "tableId": "locomotion_table",
            "layout": "full",
        },
        {
            "id": "bend_heading",
            "type": "markdown",
            "sourceId": "analysis",
            "layout": "full",
            "body": (
                "## 这条数据只能评估轻微弯腰，且三种方法都明显欠跟踪\n\n"
                "记录中的腰 pitch 峰值只有 -6.32°，不代表深度弯腰。三种 replay 的"
                "实际峰值都只有约 -1.8°，腰 pitch MAE 都约 4.5°，方法间差别小于 "
                "0.04°。这更像 policy 对该动作的跟踪限制，而不是 20 Hz 下采样方法"
                "造成的差异。"
            ),
        },
        {
            "id": "bend_table_block",
            "type": "table",
            "tableId": "bend_table",
            "layout": "full",
        },
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "analysis",
            "layout": "full",
            "body": (
                "## 分析范围与指标定义\n\n"
                "共同时间窗为 1,421 帧、28.4 s、50 Hz。片段包括：直线行走 "
                "1.2–7.8 s（331 帧）、原地动作 8.8–19.8 s（551 帧）、全身转向 "
                "20.4–24.6 s（211 帧），以及从原地动作中筛出的轻微弯腰 319 帧。\n\n"
                "- **重建误差**：20 Hz 文件经现有 loader 上采样、首帧对齐和 EMA 后，"
                "相对直接 50 Hz 参考的误差。\n"
                "- **端到端误差**：MuJoCo 实际状态相对直接 50 Hz 参考的误差，"
                "同时包含输入失真和 policy 跟踪误差。\n"
                "- **直走完成率**：实际 XY 位移 / 参考 XY 位移；横向 RMS 是相对"
                "参考直线的垂向距离。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "analysis",
            "layout": "full",
            "body": (
                "## 方法复现的是当前真实 replay 链路\n\n"
                "三条 NPZ 均通过 `load_offline_motions`，使用同一首帧 SE(2) XY/yaw "
                "对齐、EMA α=0.8、`qpos2kpt` 转换、HGPT tracking ONNX 和 50 Hz "
                "headless MuJoCo。policy 使用 CPUExecutionProvider；该仿真是确定性的，"
                "因此重复同一输入不会产生随机方差。比较只截取三条轨迹共同覆盖的"
                "核心帧，并排除 loader 添加的前 0.5 s 过渡。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "analysis",
            "layout": "full",
            "body": (
                "## 限制与稳健性\n\n"
                "这是单条约 28 s 的确定性仿真，不能给出跨操作者、跨速度或真机地面"
                "条件的置信区间。片段边界根据本条轨迹人工设定；轻微弯腰与原地动作"
                "存在重叠。绝对 root XY 误差会累积前序行走的欠跟踪，因此动作段主要"
                "应看关节/KPT，而不是该段的绝对 XY。当前结论反映“现有 loader + "
                "现有 EMA + HGPT policy”的整体效果；它不能单独归因于采样算法。"
            ),
        },
        {
            "id": "recommendation",
            "type": "markdown",
            "sourceId": "analysis",
            "layout": "full",
            "body": (
                "## 建议：保留 50 Hz master；20 Hz 离线副本用 interpolation\n\n"
                "1. **用于 HGPT replay 的主数据保留 50 Hz。** 这是直走、动作和转向"
                "综合误差最低的方案。\n"
                "2. **如果因存储或训练接口必须生成 20 Hz，离线副本选 interpolation。** "
                "保留 50 Hz master，避免不可逆地丢掉高频信息。\n"
                "3. **causal-latest 只用于在线流。** 它不需要未来帧，但离线 replay "
                "没有必要为此牺牲保真度。\n"
                "4. **下一步应把 EMA 改成按时间常数配置。** 若希望 20 Hz 与 50 Hz "
                "公平比较，20 Hz 的 α 应约为 0.572，而不是继续使用 0.8。"
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 仍需回答的问题\n\n"
                "- 在 20 Hz 使用时间常数匹配的 EMA（α≈0.572）后，直走完成率能恢复多少？\n"
                "- 真机上 causal 与 interpolation 的差异是否仍小于 DDS、相机和地面扰动？\n"
                "- 一条专门包含深度弯腰、快速转身和手臂快速动作的轨迹，是否会放大"
                "两种 20 Hz 方法的差异？"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "HGPT 20 Hz 下采样与 50 Hz Replay 定量对比",
            "description": (
                "两种 20 Hz 下采样/上采样方法与直接 50 Hz replay 的 HGPT+MuJoCo "
                "端到端误差比较。"
            ),
            "generatedAt": generated_at,
            "blocks": blocks,
            "charts": [
                {
                    "id": "tracking_joint_chart",
                    "title": "端到端 29 关节位置 MAE",
                    "subtitle": "MuJoCo 实际 qpos 相对直接 50 Hz 参考；越低越好",
                    "type": "bar",
                    "dataset": "tracking",
                    "sourceId": "tracking_source",
                    "encodings": {
                        "x": {
                            "field": "segment_label",
                            "type": "nominal",
                            "label": "动作片段",
                        },
                        "y": {
                            "field": "e2e_joint_mae_deg",
                            "type": "quantitative",
                            "label": "关节 MAE",
                            "unit": "deg",
                        },
                        "color": {
                            "field": "method_label",
                            "type": "nominal",
                            "label": "Replay 方法",
                        },
                    },
                    "unit": "deg",
                    "layout": "full",
                },
                {
                    "id": "reconstruction_joint_chart",
                    "title": "20 Hz 参考重建的 29 关节位置 MAE",
                    "subtitle": "输入 policy 前相对直接 50 Hz 参考；越低越好",
                    "type": "bar",
                    "dataset": "reconstruction",
                    "sourceId": "reconstruction_source",
                    "encodings": {
                        "x": {
                            "field": "segment_label",
                            "type": "nominal",
                            "label": "动作片段",
                        },
                        "y": {
                            "field": "joint_mae_deg",
                            "type": "quantitative",
                            "label": "关节 MAE",
                            "unit": "deg",
                        },
                        "color": {
                            "field": "method_label",
                            "type": "nominal",
                            "label": "20 Hz 方法",
                        },
                    },
                    "unit": "deg",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "locomotion_table",
                    "title": "直走与全身转向指标",
                    "subtitle": "直走参考 3.42 m；转向参考 excursion 90.82°",
                    "dataset": "locomotion",
                    "sourceId": "locomotion_source",
                    "defaultSort": {
                        "field": "straight_completion_pct",
                        "direction": "desc",
                    },
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "method", "label": "方法", "type": "text"},
                        {
                            "field": "straight_completion_pct",
                            "label": "直走完成率 (%)",
                            "format": "number",
                        },
                        {
                            "field": "straight_cross_track_rms_mm",
                            "label": "横向 RMS (mm)",
                            "format": "number",
                        },
                        {
                            "field": "straight_heading_error_deg",
                            "label": "航向误差 (°)",
                            "format": "number",
                        },
                        {
                            "field": "turn_actual_excursion_deg",
                            "label": "实际转向幅度 (°)",
                            "format": "number",
                        },
                        {
                            "field": "turn_yaw_mae_deg",
                            "label": "转向 yaw MAE (°)",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "bend_table",
                    "title": "轻微弯腰的腰 pitch 跟踪",
                    "subtitle": "仅 319 帧轻微弯腰 proxy；不代表深度弯腰",
                    "dataset": "bend",
                    "sourceId": "bend_source",
                    "defaultSort": {"field": "waist_pitch_mae_deg", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "method", "label": "方法", "type": "text"},
                        {
                            "field": "reference_peak_deg",
                            "label": "参考峰值 (°)",
                            "format": "number",
                        },
                        {
                            "field": "actual_peak_deg",
                            "label": "实际峰值 (°)",
                            "format": "number",
                        },
                        {
                            "field": "waist_pitch_mae_deg",
                            "label": "腰 pitch MAE (°)",
                            "format": "number",
                        },
                    ],
                },
            ],
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "reconstruction": reconstruction,
                "tracking": tracking,
                "locomotion": locomotion,
                "bend": bend,
            },
        },
        "sources": sources,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("reports/downsampling_replay_analysis/analysis.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/downsampling_replay_analysis/artifact.json"),
    )
    args = parser.parse_args()
    build_artifact(args.analysis, args.output)


if __name__ == "__main__":
    main()
