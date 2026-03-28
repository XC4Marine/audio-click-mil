

## Prior-Guided Attention for Weakly Supervised Acoustic Event Detection and Localization in Marine Passive Acoustic Monitoring

#### 输入文件
音频来自同一设备、同一海域、不存在domain shift。

文件根目录："D:\Project_Github\audio_click_mil\"
original_audio_dir: "data\original_audio"，original_audio中有35个.wav文件，每个文件的最后两位数是Ori_file_num(No.)(e.g., _01.wav)。

读取original_audio_dir中的data\BurstPulseTrains.csv、data\BuzzTrains.csv、data\ClickTrains.csv三个csv文件中的Train_start(s)、Train_end(s)和Ori_file_num(No.)列
```
文件格式：
Train_num(No.),Ori_file_num(No.),Train_start(s),Train_end(s),Train_duration(ms),Num_of_pulses,Mean_SNR(dB),Mean_IPI(ms),Mean_SPLpp(dB),Mean_Duration(μs),Mean_Fpeak(kHz),Mean_Bandwidth_-3dB(kHz),Mean_Bandwidth_-10dB(kHz)
1,4,1591.6738,1591.7458,72.03,21,29.3,3.59,140.32,97.64,43.21,12.83,50.54
```

### 1 Introduction （写作建议）
##### 第一段：[PAM + 生态意义（引出应用场景）]
PAM重要性、时间分布的意义、Click信号的生态意义
##### 第二段和第三段：[弱标签 + 方法问题（收回到信号处理）]
把生态问题转成弱监督问题-现有方法不足（MIL+attention，不稳定）
##### 第四段[解决方案]干净利落
引入先验
##### 第五段 生态意义
生态意义点到为止
##### 主要贡献
1.把“先验”引入attention机制

2.在没有强标签的情况下，实现并评估了定位能力

3.方法可以用于时间分布分析。“我没研究生态，但我让生态研究变得可行”。

1.真实情景下大部分标注的海洋哺乳动物PAM数据，是弱标签。强标签（秒级的定位）成本高昂，针对特定项目。为简化打弱标签的流程，使用MIL。为充分利用弱标签所隐含的信息，使用attention-based的MIL，获取信号定位，为进一步的物种识别、强标签获取提供基础。
2.用能量、脉冲数量先验的方式，引导注意力机制关注真正信号位置。加快收敛速度、提升模型可解释性。
3.对中华白海豚，Whistle信号是个体识别标签，不易采集到，且多用于个体识别。pulse信号通常用于检测中华白海豚的有无。因此采用ClickTrain、buzztrain、bursttrain信息作为包级标签依据。
4.验证该方法在辅助物种调查的能力——是否能描绘出物种时间分布?结合注意力分数来描述物种活跃度。


### 2 Method
#### 2.1 数据准备与划分：
已完成1：依据三个标注文件（BurstPulseTrains.csv、BuzzTrains.csv、ClickTrains.csv）中的 Train_start(s)、Train_end(s) 和 Ori_file_num 列。本研究视三者为广义 pulse瞬态事件，未作细分类。音频12为全噪声文件，直接剔除，剩余34个文件。利用.csv文件先给音频划分标签：每个audiofile被划分为1min的包，包含Pulse信号的任意包均标注为正包。得到正包数量317。选取319个负包，让正负包样本大致平衡。同时计算了每个正包中，噪声的持续百分比，范围为63.56%-99.92%。

代码：scripts\01_prepare_dataset.ipynb

输出：

（1）processed_data\balanced_bag_labels.csv
```
数据格式：
file_num,audio_path,bag_idx,bag_start_sec,bag_end_sec,bag_label,total_train_duration_sec,bag_duration_sec,signal_ratio,noise_ratio
1,D:\Project_Github\audio_click_mil\data\original_audio\Ori_Recording_01.wav,4,240.0,300.0,1,1.0277,60.0,0.017128333333333336,0.9828716666666667
```
（2）processed_data\noise_stats.txt

已完成2：后续模型实验将进行10折交叉验证，在这一步划分数据：固定随机种子，随机划分34个文件为10折，训练fold中80%音频用于训练，20%音频用于验证。每一折，训练、验证、测试的audiofile的配置，放在一个.json文件中。

代码：\scripts\02_10fold_divide.ipynb

输出：processed_data\folds中的十个.json文件
```
{
  "fold": 0,
  "test_files": [6,21,22],
  "train_files": ["<file_id_1>","<file_id_2>","..."],
  "val_files": ["<file_id_1>","<file_id_2>","..."]
}
```

#### 2.2 实例级先验特征构建
已完成1：每个audiofile音频先进行5kHz-200kHz的滤波，再下采样为200kHz。按照processed_data\balanced_bag_labels.csv中的配置，每个1分钟的bag被均匀划分为60个1秒实例（无重叠，不足1min的部分补零），将60个实例堆叠后形成一个包的.npy表示。每个 .npy 形状: (60, 200000)。

代码：scripts\03_instance_bag_npy.ipynb

输出：D:\Project_Github\audio_click_mil\processed_data\instances
```
file_01_bag_000_label_0.npy
命名方式：file_<audio序号数>_bag_<在一个audio中的位置>_label_<标签>.npy
...
```

2.建立一个简单的attention-based MIL框架：
使用processed_data\folds\fold_0.json的配置，根据instances文件夹中.npy文件的audio序号数，找到属于集合的包文件。特征提取采用64阶的log-mel图，使用一个简单的CNN，将log-mel特征压缩到n维,每个包表示成(60,n)的形式。评估指标为Precision、Recall、Accuracy。

同时进行Instance-level定位能力的比较：
* 对每个正样本 bag，设定固定注意力阈值 $  \theta  $（需报告敏感性分析），将 $  a_i > \theta  $ 的实例视为模型预测的信号位置（记为检测集合 $  D_i  $）；画出FDR-Recall图来确定最好的注意力阈值，位于图右下角的值是最好的。
* 取每个预测实例的中心时间戳作为定位点；
* 地面真值（ground truth）为标注文件中所有 train 片段的中心时间戳（记为 $  G_i  $）；
采用容忍窗口1780ms，允许一个 ground truth 被多个检测命中，使用 many-to-many 匹配方式计算FDR、Recall和F1：
$$\text{Precision} = \frac{\sum_i \text{card}\{t_d \in D_i \mid \exists g_k \in G_i, |t_d-g_k|\le\tau\}}{\sum_i \text{card}\{D_i\}}$$
$$\text{Recall} = \frac{\sum_i \text{card}\{g_k \in G_i \mid \exists t_d \in D_i, |t_d-g_k|\le\tau\}}{\sum_i \text{card}\{G_i\}}$$
$$\text{FDR} = 1 - \text{Precision}$$




## Grok，请先忽略以下内容，只要阅读以上内容


为增强弱监督条件下对pulse瞬态信号的定位能力，我们为每个实例提取能量先验特征：

瞬态能量强度：计算Teager-Kaiser 能量算子在该 1秒内的最大值，因为pulse是瞬态冲击（短时能量突变），最大值能最敏锐地抓住这个“峰值涌现。而总能量值会被全秒的背景噪声淹没。平均值会被稀释。

该特征分别进行per-file内 min-max 归一化后，得到每个实例的先验分数 $  s_i \in [0,1]  $，用于表征该时间片段含有pulse信号的先验置信度。



#### 2.3 模型架构与先验引导注意力机制
所有音频首先进行预处理：

* 降采样至 48 kHz；
* 应用 5 kHz 高通巴特沃斯滤波器（6阶）去除低频噪声；
* 以 1分钟 为单位均匀切分为非重叠的 bag（每个 bag 包含 60 个 1秒实例）。考虑到弱监督下 bag 粒度已较粗，因此采用任何重叠即正的保守策略。
* 将每个bag划分为60个1s的instance。

* 特征提取：对每个instance做40维MFCC（保存.npy）  → 每个bag的60个instance MFCC输入Temporal Convolutional Network (TCN) [@fonollosa2025Temporal]得到特征表示；

* 注意力模块：在标准注意力 logit 计算中引入先验引导：$$a_i = \text{softmax}\left( tanh(f(\mathbf{x}_i)) + \alpha \cdot g(s_i) \right)$$其中 $  f(\mathbf{x}_i)  $ 为模型学习得到的实例判别分数，$  g(\cdot)  $ 为先验分数的映射函数（$linear(s_i) → ReLU → linear$），$  \alpha  $ 为固定的超参数，用于控制先验影响强度；
* Bag 表示：对所有实例特征进行注意力加权求和，得到 bag-level 表示；
* 分类头：二分类（sigmoid 输出 bag 是否含有 pulse）。

#### 2.4 训练设置

采用 5 折交叉验证，每折独立训练模型。损失函数为标准二元交叉熵（可结合文献[@nihal2025Weakly]中推荐的弱监督稳定技巧）。训练完成后，将 5 个模型分别应用于完整测试集（音频1–8）的 1分钟 bag，得到5组预测结果用于后续统计分析（报告均值±标准差）。

### 3 Result
#### 任务1. 检测性能（Bag-level Classification）
评估指标主要包括：

* Recall（避免漏检关键生态事件）；
* False Discovery Rate (FDR = 1 − Precision）（控制误报率）；
* Accuracy（整体正确率，作为参考）。

在完整测试集（音频1–8）上报告 5 个模型的平均性能及标准差。[@fonollosa2024Comparing]

#### 任务2. 定位性能（Instance-level Localization under Weak Supervision）
[@nihal2025Weakly]
利用注意力权重实现弱监督定位评估。流程如下：

* 对每个正样本 bag，设定固定注意力阈值 $  \theta  $（需报告敏感性分析），将 $  a_i > \theta  $ 的实例视为模型预测的信号位置（记为检测集合 $  D_i  $）；画出FDR-Recall图来确定最好的注意力阈值，位于图右下角的值是最好的。
* 取每个预测实例的中心时间戳作为定位点；
* 地面真值（ground truth）为标注文件中所有 train 片段的中心时间戳（记为 $  G_i  $）；
采用容忍窗口 $  \tau  $（建议 0.5s、1s、2s 多值报告），允许一个 ground truth 被多个检测命中，使用 many-to-many 匹配方式计算FDR、Recall和F1：
$$\text{Precision} = \frac{\sum_i \text{card}\{t_d \in D_i \mid \exists g_k \in G_i, |t_d-g_k|\le\tau\}}{\sum_i \text{card}\{D_i\}}$$
$$\text{Recall} = \frac{\sum_i \text{card}\{g_k \in G_i \mid \exists t_d \in D_i, |t_d-g_k|\le\tau\}}{\sum_i \text{card}\{G_i\}}$$
$$\text{FDR} = 1 - \text{Precision}$$
容忍窗口采用训练集基于真实 train 时长分布的中位数百分位1/5，2/5，80%。这种基于真实信号时长分布的选择，既避免了因 $  \tau  $ 过大导致的虚假高 Precision，也保证了即使存在轻微的时间对齐误差（例如标注边界模糊或模型实例中心偏移）仍能被合理匹配，从而使定位评估更贴近实际生态应用场景。

#### 任务3 消融实验
由于instance level缺少监督，其可能不是真的在定位pulse，而是学到了某种统计偏好。设计以下三组对比，验证先验引导的有效性与物理合理性：

1. Baseline：标准 Attention-based MIL（无先验）；
2. $+$ Energy：引入能量先验；
3. α 敏感性分析：固定先验特征，改变 $  \alpha \in \{0, 0.5, 1.0, 2.0\}  $，记录收敛 epoch、检测性能及定位 Precision/Recall

对比指标：bag-level Recall / FDR + instance-level Precision / Recall
#### 任务4 Ecological application （加分项）. 
[@fonollosa2024Comparing]
在完整测试集上执行以下递进分析：

* 1.叫声计数趋势比较

统计 5 个模型在每条原始音频（1–8）中检测到的正样本 bag 数量，与人工标注的 pulse train 数量数量进行比较。绘制箱型图 + 折线图，计算 Pearson 相关系数 评估趋势一致性。做显著性检验（p-value）。

* 2.声学活动指数（AAI）比较

采用概率累加方式计算传统 AAI（每个 bag 的正样本概率之和），与真实叫声趋势对比，报告 Pearson 相关系数。做显著性检验（p-value）。

* 3.Attention-weighted Activity Index（本文提出）

为每个 1分钟 bag 计算：$$\text{AWAI} = P_{\text{bag}} \times \sum_{i:\ a_i > 0.05} a_i \times \Delta t$$（其中 $  \Delta t = 1  $ 秒，仅累加高置信实例的注意力权重和，再乘以 bag 概率，得到“等效高置信pulse活动秒数”）。
与真实 train 总时长比较，计算 Pearson 相关系数。$\alpha_i$的取值要做敏感性分析，通过Pearson 相关系数来确定。做显著性检验（p-value）。

通过以上三种指标的层层递进对比，展示先验引导注意力机制在弱监督条件下不仅提升检测与定位性能，还能为海洋哺乳动物声学活动的时间分布、昼夜节律及季节模式分析提供更平滑、更有生态解释力的量化依据。（写论文时，加入三种指标的可比性说明。）






