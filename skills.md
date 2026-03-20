

# CWD-MIL Hierarchical Weakly-Supervised Framework

**作者**：Grok（基于用户需求与Fu et al. 2025数据集）
**项目背景**：  
**项目目标**：使用中华白海豚（Indo-Pacific Humpback Dolphin）声学数据集，实现**纯弱监督**的多类型声信号检测（哨声 + 脉冲串）、时序定位与行为洞见分析。  
**核心创新**：分层Attention MIL（Bag-level存在检测 + Instance-level类型细分 + 元数据融合），训练时**只使用bag-level 0/1标签**，评估时利用精确时间戳。

## 1. 数据集完整结构（必须严格遵守）

### data/ 文件夹内容（必须包含原始音频）

data/
├── original_audio/                  # ← 原始音频文件夹（必须放入）
│   ├── Ori_Recording_01.wav
│   ├── Ori_Recording_02.wav
│   ├── Ori_Recording_03.wav
│   └── ... (共35个录音文件)
├── WhistleParameters-clean.csv      # 哨声实例级标注（精确开始时间 + 6类Type + 参数）
├── ClickTrains.csv                  # Click脉冲串实例级标注（Train_start/Train_end + 参数）
├── BurstPulseTrains.csv             # Burst脉冲串实例级标注
├── BuzzTrains.csv                   # Buzz脉冲串实例级标注
└── Results.csv                      # 35条录音摘要（水深、群大小、行为、Pulse_train_num、Whistle_num）

**关键音频参数**（已确认）：
- 原始采样率：576 kHz（单声道 mono）
- 处理时统一下采样到48 kHz（防止计算复杂度过高，configs/default.yaml中可配置）

**弱标签生成规则**（Phase 0核心）：
- whistle_label = 1 if 该bag内存在任何哨声（通过Whistle Begins时间判断） else 0
- pulse_label = 1 if 该bag内存在任何click/burst/buzz train（通过Train_start时间判断） else 0
- 训练时**永远只用这两个0/1标签**（纯弱监督）
- 评估时使用精确的Whistle Begins / Train_start计算localization precision

## 2. 整体Pipeline（6个Phase递进式开发）

mermaid
flowchart TD
    A[data/original_audio + 4个CSV] --> B[Phase0: Bag生成 + 弱标签]
    B --> C[Phase1: 基础Bag检测]
    C --> D[Phase2: Attention定位]
    D --> E[Phase3: 分层类型细分]
    E --> F[Phase4: 元数据融合 + 行为洞见]
    F --> G[Phase5: 完整4-loss + 最终评估]
    G --> H[Phase6: GitHub打包 + 论文材料]


## 3. 每个Phase详细规范（输入/输出/模块）

### Phase 0: 数据准备（必须先跑）
**输入**：
- data/original_audio/*.wav（Ori_Recording_01.wav 等，576 kHz单声道）
- WhistleParameters-clean.csv
- ClickTrains.csv
- BurstPulseTrains.csv
- BuzzTrains.csv
- Results.csv

**输出**：
- `data/bags.pkl`（list of dict：{'bag_id', 'instances': tensor[M, 522], 'label': [whistle, pulse], 'meta': {...}, 'start_time', 'end_time'})
- `results/phase0/dataset_summary.csv`（Table 1：总哨声100、高质量100、clicks832、burst15、buzz50、水深范围、行为分布）
- `results/phase0/dataset_report.txt`

- **核心逻辑**：读取配置，遍历所有原始音频和标注CSV文件。对每个音频文件，先下采样，再按固定时长（如10秒）切分为实例。然后基于标注的时间戳，为每个bag（由多个连续实例组成）生成弱监督标签（是否有哨声/脉冲）。最后提取522维特征并保存到`bags.pkl`。同时生成数据摘要。

### Phase 1: 基础Bag-level检测
**输入**：Phase0的bags.pkl
**输出**：
- `results/phase1/performance_table.csv`（Table 2）
- `results/phase1/fig1_f1_vs_bag_length.png`
- `results/phase1/baglevel_report.txt`
- **核心逻辑**：加载`bags.pkl`，根据配置（是否交叉验证）划分训练/验证/测试集。初始化`SimpleMLP`模型，使用`FocalBCE`损失，进行模型训练与验证。最后在测试集上评估袋级别检测性能（F1, Precision, Recall），并绘制结果图表。


### Phase 2: Instance-level Attention定位
**输入**：Phase0 bags
**输出**：
- `results/phase2/localization_table.csv`（Table 3）
- `results/phase2/fig2_attention_heatmap_examples.png`
- `results/phase2/instancelevel_report.txt`

**脚本核心逻辑**：在Phase1数据集划分基础上，训练带`AttentionModule`的模型。总损失包含bag级别的`FocalBCE`和instance级别的`sparsity_loss` + `temporal_smoothness_loss`。训练后，通过分析高权重的实例，将其中心时间与真实标注时间戳比对（容忍2秒误差），计算定位精度，并可视化注意力热图。


### Phase 3: 分层类型细分
**输入**：Phase0 + WhistleParameters-clean.csv（Type列）+ Pulse trains子类
**输出**：
- `results/phase3/type_performance.csv`（Table 4）
- `results/phase3/confusion_matrix.png`（Figure 3）
- `results/phase3/ablation_table.csv`（Table 5）
- `results/phase3/performance_report.txt`

**脚本核心逻辑**：扩展模型，增加用于哨声6分类和脉冲3分类的`type_head`。在训练时，对于已知存在信号类型的bag，其内部实例的类型伪标签用于计算`type_focal_loss`（采用类别权重）。评估时，在定位出的实例上评估细分类别精度，并进行消融实验（如关闭类型头训练）。


### Phase 4: 元数据融合（行为洞见）
**输入**：Results.csv（水深/群大小/行为）
**输出**：
- `results/phase4/behavior_correlation.png`（Figure 4）
- `results/phase4/density_trend.csv`（Table 6）
- `results/phase4/fusion_report.txt`

**脚本核心逻辑**：在数据集中集成`MetadataEmbedder`，将元数据（水深、群大小、季节、行为类别）映射为32维向量，与音频特征拼接后输入模型。训练融合模型后，运行`scripts/behavior_analysis.py`分析模型预测（如信号检出率、类型分布）与不同行为、水深等元数据的相关性，并可视化。


### Phase 5: 完整损失 + 最终评估
**输入**：前4个Phase模型
**输出**：
- `results/phase5/final_summary.csv`（Table 7）
- `results/phase5/loss_curves.png`（Figure 5）
- `results/phase5/evaluation_report.txt`

**脚本核心逻辑**：使用完整的4-loss组合（bag-level FocalBCE + instance-level 稀疏/平滑损失 + 类型损失）训练最终模型。在独立的测试集上进行综合评估，包括袋级别检测、时序定位精度、类型细分精度，并汇总所有Phase的关键指标，生成最终报告。


### Phase 6: GitHub打包
**输出**：
- 完整仓库结构（见下方）
- `README.md`（一键复现 + 论文图表说明）
- `Supplementary_Table_S1.csv`

## 4. 关键技术细节（已固定，不可随意更改）
- **包（bag）定义**固定1min非重叠音频片段
- **实例（Instance）定义**：固定10秒非重叠音频片段（可在configs/default.yaml中调整为5–15秒）。每个bag（1min）按此长度均匀切分，每个实例对应连续的10秒时间窗口。
- **特征维度（feat_dim）**：522维。在Phase 0的prepare_bags.py中一次性提取并保存到bags.pkl，包括128维Mel频谱图（20Hz–150kHz，下采样后48kHz处理）的mean/std/max/min四个统计值（共512维） + 10个时间域统计特征（RMS、ZCR、PeakAmp、Mean_IPI、TEKO均值、FDR均值、EnvStats均值/方差、TempCentroid等）。
- **AttentionModule**：经典MIL注意力结构，W为[522, 128]可学习矩阵，v为[128]向量，通过v^T * tanh(W * feat + b)计算分数后softmax得到alphas。
- **SimpleMLP特征提取器**：3层全连接层，每层隐藏单元256 + ReLU激活。
- **MetadataEmbedder**：两层MLP，将水深/群大小/季节/行为4个标量映射到32维嵌入后与音频特征拼接。
- **数据分割**：严格按Ori_file_num（35个.wav文件，每个文件30mins）划分，70%训练 / 15%验证 / 15%测试，确保同一录音的所有bag只出现在同一个集合中（blocked CV）。当启用k折交叉验证时，按录音文件分组进行k折划分。
- **Localization precision**：每个高attention权重的10秒实例中心时间与真实标注（Whistle Begins / Train_start）比较，绝对差≤2秒视为正确匹配，precision = 正确匹配数 / 总预测检测数。Phase 2评估完全在时间戳级别（实例开始时间 + 偏移）。
- **超参数（固定默认值，可通过configs/default.yaml微调）**：
  - 初始学习率：1e-4
  - 优化器：AdamW（weight_decay=0.01，gradient_clip=1.0）
  - Batch Size：32（RTX 4060稳定）
  - Epochs：80 + 早停（patience=15）
  - Focal Loss：gamma=2.0，alpha=0.25

## 5. 配置管理（configs/default.yaml）

所有可调参数统一放在configs/default.yaml中（Phase 0开始读取）：
```yaml
data:
  root: "data"
  original_audio_dir: "original_audio"
  downsample_rate: 48000          # ← 新增：强制下采样到48kHz
  instance_duration_sec: 10       # 可调5-15
  bag_duration_target_sec: 180
  bag_hop_sec: 60

model:
  feat_dim: 522
  hidden_dim: 256
  meta_embed_dim: 32
  attention_hidden: 128

train:
  lr: 0.0001
  optimizer: "AdamW"
  weight_decay: 0.01
  batch_size: 32
  epochs: 80
  early_stop_patience: 15
  focal_gamma: 2.0
  focal_alpha: 0.25
  localization_tolerance_sec: 2.0
  # --- 新增：实验配置接口 ---
  cross_validation: false         # true: 进行k折交叉验证 | false: 使用固定70/15/15划分
  n_folds: 10                     # 交叉验证折数

phase:
  current: 0
```

## 6. 代码模块输入输出规范（完整列表）

### 编码规范
实现各模块代码时，应遵循简洁、高效的原则，避免冗余代码和复杂嵌套，确保逻辑清晰。并告诉我如何运行。

## 7. GitHub仓库完整结构（必须严格遵守）
项目根目录：D:\Project_Github\audio_click_mil

audio_click_mil/
├── configs/
│   └── default.yaml
├── data/
│   ├── original_audio/              # ← 必须放入所有Ori_Recording_*.wav
│   ├── WhistleParameters-clean.csv
│   ├── ClickTrains.csv
│   ├── BurstPulseTrains.csv
│   ├── BuzzTrains.csv
│   └── Results.csv
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   └── evaluate.py           # 评估函数库
├── scripts/
│   ├── prepare_bags.py        # Phase 0
│   ├── phase1_baseline.py     # Phase 1
│   ├── phase2_attention_loc.py # Phase 2
│   ├── phase3_type_finegrained.py # Phase 3
│   ├── phase4_metadata_fusion.py  # Phase 4
│   ├── phase5_full_model.py    # Phase 5
│   └── behavior_analysis.py    # 辅助分析
├── notebooks/
│   ├── Phase0_DataPrep.ipynb   # 用于数据探索和特征可视化
│   ├── Phase1_Baseline.ipynb   # 用于基线结果分析和调试
│   └── Full_Experiment.ipynb   # 用于最终结果整合与高级可视化
├── results/                         # 所有Phase输出
├── README.md                        # 一键复现 + 论文图表映射
├── requirements.txt
├── .gitignore
└── skill.md                         # 本文件（LLM理解文档）


## 8. 运行命令（一键复现）

bash
Phase 0（必须先跑）：数据预处理与特征提取

python scripts/prepare_bags.py

Phase 1-5（可按需运行，也可按顺序全部运行以复现完整实验）

python scripts/phase1_baseline.py
python scripts/phase2_attention_loc.py
python scripts/phase3_type_finegrained.py
python scripts/phase4_metadata_fusion.py
python scripts/phase5_full_model.py

启用10折交叉验证运行某个Phase（需在configs/default.yaml中设置 cross_validation: true）

python scripts/phase1_baseline.py


## 9. 论文图表映射（直接复制）

- Table 1 ← Phase0 summary.csv
- Figure 1 ← Phase1
- Figure 2 ← Phase2
- Table 5（消融）← Phase3
- Figure 4（行为）← Phase4
- Table 7（最终）← Phase5


## CSV 数据结构示例（关键列 + 真实样例行）

**Results.csv**（35条录音摘要）
关键列：Data (YYYY-MM-DD), Season, Original audio_file (No.), Start time, End time, Pulse_train_num, Whistle_num, Water Depth (m), Dolphin group size, Dolphin Behavior
样例行：

2022/6/20,Summer,1,9:59:58,10:29:57,"18 (Click:17; BurstPulse:1; Buzz:0)","1 (High quality:1)",13.9,5,Travelling
2022/12/4,Winter,6,10:06:19,10:36:18,"104 (Click:57; BurstPulse:4; Buzz:43)","28 (High quality:19)",7.4,6,Travelling


**WhistleParameters-clean.csv**（哨声实例级标注）
关键列：Whistle Number, Original Audio File, Whistle Begins (s), Type（6类）, Quality Grade, Duration (ms), Start Frequency (kHz), End Frequency (kHz), ...
样例行：

1,1,1758.166,Sinusoidal,3,211.8,4.67,4.93,...
5,5,1124.541,Upsweep,2,142.6,3.56,10.89,...


**ClickTrains.csv**（Click脉冲串实例级标注）
关键列：Train_num, Ori_file_num, Train_start(s), Train_end(s), Train_duration(ms), Num_of_pulses, Mean_IPI(ms), Mean_Fpeak(kHz), ...
样例行：

1,1,278.2543,278.7155,461.2,4,152.99,58.33,...
10,1,1000.4539,1001.0202,566.3,15,40.25,113.47,...


**BurstPulseTrains.csv**（Burst脉冲串实例级标注）
关键列：Train_num, Ori_file_num, Train_start(s), Train_end(s), Mean_IPI(ms), Mean_Fpeak(kHz), ...
样例行：

1,1,1631.0973,1631.1532,55.86,8,7.93,39.38,...
8,10,351.6915,352.7461,1054.68,75,14.25,74.78,...


**BuzzTrains.csv**（Buzz脉冲串实例级标注）
关键列：Train_num, Ori_file_num, Train_start(s), Train_end(s), Mean_IPI(ms), Mean_Fpeak(kHz), ...
样例行：

1,4,1591.6738,1591.7458,72.03,21,3.59,43.21,...
24,6,1165.5617,1166.0812,519.45,167,3.1,91.06,...


**文件结束**
（此skill.md已完整集成所有要求，包括48kHz下采样、522维特征、10秒实例、Attention/MLP/Metadata细节、数据分割、localization 2秒容忍、超参数默认值、configs/default.yaml字段、脚本化执行及k折交叉验证配置接口等。）