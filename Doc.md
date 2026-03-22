

## Prior-Guided Attention for Weakly Supervised Acoustic Event Detection and Localization in Marine Passive Acoustic Monitoring

### 1 Introduction
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
3.对中华白海豚，Whistle信号是个体识别标签，不易采集到，且多用于个体识别。Click信号通常用于检测中华白海豚的有无。因此采用ClickTrain、buzztrain、bursttrain信息作为包级标签依据。
4.验证该方法在辅助物种调查的能力——是否能描绘出物种时间分布?结合注意力分数来描述物种活跃度。


### 2 Method
#### 数据处理：
1.由于音频1-音频8体现了完整的四季周期，可以用于验证叫声检测模型的外推能力（若检测出的叫声数量与实际叫声熟练的季节分布趋势大致相同，则可说明问题），因此，我们将音频1-音频8作为完整测试集。音频9-音频35（音频12除外，因为其中没有click信号，都是噪声。因此剩余26条音频），作为训练集来进行5-fold交叉验证和显著性分析。将5个模型应用于音频1-音频8（以划分为1min的bag）中叫声信号的检测，得到5组检测结果，做显著性分析。将5折验证所需要的5组数据保存在不同的文件夹下。
2.先取其中一折交叉验证的训练集、测试集，将训练集中的最后一个.wav文件作为验证集。将每个.wav文件下采样到48kHz，使用5kHz的高通巴特沃斯滤波器进行滤波，切割成1min的bag。于是得到测试集、训练集、验证集的包数据。
3.根据BurstPulse、Buzz、Click在每个.wav文件中的位置和持续时间，给切割后的bag打上存在信号（1）和不存在信号（0）的标签。同时记录每一个标签为1的bag的中，信号的持续时间(s)。


#### 包级实例准备
为避免传统多实例学习（MIL）在弱监督条件下对关键实例定位能力不足的问题，同时兼顾后续声学活动趋势分析与时间分布建模的需求，本文在实例构建与注意力机制设计中引入基于瞬态特征的先验信息。

首先，对每段长度为1 min的音频数据进行均匀划分，得到固定长度为1 s的时间片段（instance），从而构建包含60个实例的bag。该划分方式在保证时间覆盖均匀性的同时，避免了基于事件检测的切分策略可能引入的时间重叠与分布偏置问题。

在实例级别，为每个时间片段提取两类表征瞬态特征的先验信息：一是基于瞬态检测算法获得的脉冲事件数量（transient count），二是基于能量算子（如短时能量或Teager–Kaiser能量算子）计算的瞬态能量强度。二者共同构成实例的先验评分函数，用于刻画该时间片段中潜在click-like信号的显著程度。通过对上述特征进行归一化与融合，得到每个实例的先验分数$s_i$ 。

在模型构建方面，采用基于注意力机制的多实例学习框架。对于每个实例$x_i$,首先通过特征提取网络获得其嵌入表示；随后，在标准注意力计算过程中引入先验分数，对注意力权重进行引导：
$$a_i = softmax( f(x_i) + α · g(s_i) )$$
其中，$f(x_i)$表示由模型学习得到的实例判别特征，$s_i$为先验评分，$g(⋅)$为归一化映射函数，$\alpha$为控制先验影响程度的权重系数。该设计使模型在弱监督条件下能够利用瞬态结构信息，对潜在信号区域赋予更高关注度，同时保留对非典型信号的自适应学习能力。

最终，通过对实例特征进行加权聚合获得bag级表示，并完成分类任务。在此基础上，模型输出的注意力权重可进一步映射回时间轴，用于刻画声学事件在时间维度上的分布情况，从而支持后续的定位能力评估与声学活动趋势分析。

#### 模型架构

MFCC + 时域特征 → TCN架构 [@fonollosa2025Temporal] → 先验的attention-based MIL（必要损失函数[@nihal2025Weakly]）

### 3 Result
#### 任务1. Detection Performance 
[Fonolossia et al. Comparing neural networks against click train detectors to reveal temporal trends in passive acoustic sperm whale detections]
在模型性能方面，主要关注两个指标，（1）Recall；（2）FDR。Recall不能低，模型不要漏检太多；FDR=1-Precision（错误的报警）不能太高，否则误检太多。同时额外关注Accuracy。[@fonollosa2024Comparing]

#### 任务2. Localization performance（🔥重点）
[@nihal2025Weakly]
通过Attention实现定位的效果，检测精度的分析：设定注意力阈值，高于阈值的instance视为该处有信号检测到。比如一个bag是30s，我们通过检测取出了其中10个概率较高的instances(t1,t2,...,t10)，通过特征提取和基于注意力机制的MIL，t1和t2的注意力高于阈值。我们先取t1和t2的中间时间戳。在这个bag中，假设有3个instances的位置，被标注为了有click，其中间时间戳分别为g1,g2,g3。我们设定容忍度$/pi$，通过many-to-many的方式，G和T中，容忍度小于$/pi$的配对，并累计配对数量。该情况下，如果t1和t2都能够完成配对，则Precision=2/2=1；如果只有t1完成配对，则Precision=1/2。用Precision和Recall：
$$
\text{Precision} = 
\frac{
  \sum_i \ \text{card}\bigl\{ t_d \in D_i \ \big|\ 
  \exists\ g_k \in G_i,\ |t_d - g_k| \le \tau \bigr\}
}{
  \sum_i \ \text{card}\{D_i\}
}
$$
$$
\text{Recall} = 
\frac{
  \sum_i \ \text{card}\bigl\{ g_k \in G_i \ \big|\ 
  \exists\ t_d \in D_i,\ |t_d - g_k| \le \tau \bigr\}
}{
  \sum_i \ \text{card}\{G_i\}
}
$$
#### 任务3 消融实验
由于instance level缺少监督，其可能不是真的在定位click，而是学到了某种统计偏好。因此该任务是验证 attention 的“物理正确性”。

关于先验的消融实验：
1. Baseline MIL（无先验）
2. $+$ energy
3. $+$ energy + transient（你的方法）
4. $\alpha$敏感性分析, 0, 0.5, 1, 2, 额外要看收敛速度

对比：对比：Recall、FDR、定位Precision/Recall
#### 任务3 Ecological application （加分项）. 
[@fonollosa2024Comparing]
（1）统计5组每组结果分别检测到的每段原音频中的叫声信号，得到该模型在每段原音频中的检测能力的显著性分析，用折现箱型图的形式画出，和原音频的真实叫声数量比较。若趋势大致相同，则可进一步进行物种时间分布分析，提取生态学意义。用Pearson相关系数计算趋势一致性。
（2）使用概率累加的方式进行（声学活动指数AAI）的比较，更加高级，和（1）对比。基于假设：在大量bag统计下：高活动 → 更多高概率bag；低活动 → 更多低概率bag，统计意义上可能成立。用Pearson相关系数计算相关性。
（3）为了更精细地刻画中华白海豚的声学活动强度，提出attention-weighted activity index作为声学活动指数（不要定义成明确的物理量）。该指标定义为每个1分钟bag的模型输出概率乘以其注意力权重之和（仅保留权重高于0.05的instance），再乘以每个instance的时长（10秒），从而将MIL的bag-level概率转化为一个具有物理意义的连续指标——即“该1分钟内相当于有多少秒的高置信click活动”。与传统二元计数相比，EAS能够同时捕捉活动的有无、强度和时域集中程度，为后续昼夜节律、季节分布及觅食热点分析提供了平滑且生态意义明确的量化依据。用Pearson相关系数计算相关性。
以上三种方法层层递进进行比较，体现MIL方法的优势。





#### 文件
音频来自同一设备、同一海域、不存在domain shift
文件根目录："D:\Project_Github\audio_click_mil\"
original_audio_dir: "data\original_audio"，original_audio中有35个.wav文件，每个文件的最后两位数是Ori_file_num(No.)(e.g., _01.wav)。
读取original_audio_dir中的data\BurstPulseTrains.csv、data\BuzzTrains.csv、data\ClickTrains.csv三个csv文件中的Train_start(s)、Train_end(s)和Ori_file_num(No.)列

