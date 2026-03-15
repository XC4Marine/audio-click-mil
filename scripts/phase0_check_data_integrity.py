# ============================================
# Phase 0 - Task 1: 数据完整性检查与基本统计
# ============================================
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─── 1. 配置路径 ──────────────────────────────────────────────────────────────
CONFIG = {
    "data_root": "D:/Project_Github/audio_click_mil/data",
    "output_dir": "D:/Project_Github/audio_click_mil/results/phase0",
    "csv_names": {
        "whistle": "WhistleParameters-clean.csv",
        "click": "ClickTrains.csv",
        "burst": "BurstPulseTrains.csv",
        "buzz": "BuzzTrains.csv",
        "results": "Results.csv",
    }
}

# 创建输出目录
output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(exist_ok=True, parents=True)

log_lines = [f"Phase 0 - 数据完整性检查 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]

# ─── 2. 检查文件是否存在 ──────────────────────────────────────────────────────
missing = []
for k, fname in CONFIG["csv_names"].items():
    p = Path(CONFIG["data_root"]) / fname
    if not p.exists():
        missing.append(str(p))
    else:
        log_lines.append(f"✓ {fname} 存在")

if missing:
    log_lines.append("\n!!! 缺失以下文件，无法继续 !!!")
    for m in missing:
        log_lines.append(f"  {m}")
    print("\n".join(log_lines))
    (output_dir / "00_file_integrity_check.txt").write_text("\n".join(log_lines), encoding='utf-8')
    raise FileNotFoundError("关键 csv 文件缺失")

log_lines.append("\n所有必须 csv 文件都存在 ✓\n")

# ─── 3. 读取所有 csv ───────────────────────────────────────────────────────────
whistle = pd.read_csv(Path(CONFIG["data_root"]) / CONFIG["csv_names"]["whistle"])
click = pd.read_csv(Path(CONFIG["data_root"]) / CONFIG["csv_names"]["click"])
burst = pd.read_csv(Path(CONFIG["data_root"]) / CONFIG["csv_names"]["burst"])
buzz = pd.read_csv(Path(CONFIG["data_root"]) / CONFIG["csv_names"]["buzz"])
results = pd.read_csv(Path(CONFIG["data_root"]) / CONFIG["csv_names"]["results"])

# ─── 4. 基本统计 ────────────────────────────────────────────────────────────────
stats = []

stats.append(f"哨声实例数量 (WhistleParameters-clean.csv) : {len(whistle):6d} 条")
stats.append(f"Click train 数量                            : {len(click):6d} 条")
stats.append(f"BurstPulse train 数量                       : {len(burst):6d} 条")
stats.append(f"Buzz train 数量                             : {len(buzz):6d} 条")
stats.append(f"Results.csv 录音文件数                      : {len(results):6d} 条")

# 脉冲总数（click + burst + buzz）
total_pulse = len(click) + len(burst) + len(buzz)
stats.append(f"脉冲串总数 (Click+Burst+Buzz)               : {total_pulse:6d} 条")

# 文件编号匹配性
ori_files_whistle = set(whistle["Original Audio File (No.)  "].dropna().astype(int))
ori_files_click = set(click["Ori_file_num(No.)"].dropna().astype(int))
ori_files_burst = set(burst["Ori_file_num(No.)"].dropna().astype(int))
ori_files_buzz = set(buzz["Ori_file_num(No.)"].dropna().astype(int))
ori_files_results = set(results["Original audio_file (No.)"].dropna().astype(int))

stats.append(f"\n哨声出现的录音文件数    : {len(ori_files_whistle)} 个")
stats.append(f"Click 出现的录音文件数  : {len(ori_files_click)} 个")
stats.append(f"Burst 出现的录音文件数  : {len(ori_files_burst)} 个")
stats.append(f"Buzz 出现的录音文件数   : {len(ori_files_buzz)} 个")
stats.append(f"Results.csv 总录音文件数: {len(ori_files_results)} 个")

# 哨声类型分布
stats.append(f"\n哨声类型分布:")
type_counts = whistle["Type"].value_counts()
for t, c in type_counts.items():
    stats.append(f"  {t}: {c} 条")

# 质量等级分布
stats.append(f"\n哨声质量等级分布:")
quality_counts = whistle["Quality Grade"].value_counts()
for q, c in quality_counts.items():
    stats.append(f"  Grade {q}: {c} 条")

log_lines.extend(stats)

# ─── 5. 保存日志 ────────────────────────────────────────────────────────────────
log_path = output_dir / "00_file_integrity_check.txt"
log_path.write_text("\n".join(log_lines), encoding='utf-8')

print("\n".join(log_lines))
print(f"\n日志已保存至：{log_path}")