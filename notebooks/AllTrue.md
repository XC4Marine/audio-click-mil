文件根目录："D:\Project_Github\audio_click_mil\"
original_audio_dir: "data\original_audio"，original_audio中有35个.wav文件，每个文件的最后两位数是Ori_file_num(No.)(e.g., _01.wav)。

数据处理：


实例划分和包级标签准备：
[Fonollosa et al. Temporal Feature Learning in Weakly Labelled Bioacoustic Cetacean Datasets via a  Variational Autoencoder and Temporal Convolutional Network: An Interdisciplinary  Approach]




由于音频1-音频8体现了完整的四季周期，可以用于验证叫声检测模型的外推能力（若检测出的叫声数量与实际叫声熟练的季节分布趋势大致相同，则可说明问题），因此，我们将音频1-音频8作为完整测试集。音频9-音频35（音频12除外，因为其中没有click信号，都是噪声。因此剩余26条音频），作为训练集来进行5-fold交叉验证和显著性分析。将5个模型应用于音频1-音频8（以划分为1min的bag）中叫声信号的检测，得到5组检测结果，做显著性分析。

任务1. [Fonolossia et al. Comparing neural networks against click train detectors to reveal temporal trends in passive acoustic sperm whale detections]
在模型性能方面，主要关注两个指标，（1）Recall；（2）FDR。Recall不能低，模型不要漏检太多；FDR=1-Precision（错误的报警）不能太高，否则误检太多。同时额外关注Accuracy。

任务2. [Nihal et al. Weakly supervised multiple instance learning for whale call detection and temporal localization in long-duration passive acoustic monitoring]
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
任务3. [Fonolossia et al. Comparing neural networks against click train detectors to reveal temporal trends in passive acoustic sperm whale detections]
（1）统计5组每组结果分别检测到的每段原音频中的叫声信号，得到该模型在每段原音频中的检测能力的显著性分析，用折现箱型图的形式画出，和原音频的真实叫声数量比较。若趋势大致相同，则可进一步进行物种时间分布分析，提取生态学意义。用Pearson相关系数计算趋势一致性。
（2）使用概率累加的方式进行趋势的比较，更加高级，和（1）对比。基于假设：在大量bag统计下：高活动 → 更多高概率bag；低活动 → 更多低概率bag，统计意义上可能成立。

读取original_audio_dir中的data\BurstPulseTrains.csv、data\BuzzTrains.csv、data\ClickTrains.csv三个csv文件中的Train_start(s)、Train_end(s)和Ori_file_num(No.)列，找到每段音频中


每段音频按时间8：2分成训练集、测试集。
将每个.wav文件下采样到48kHz，切割成1min的bag。
根据BurstPulse、Buzz、Click在每个.wav文件中的位置，给切割后的bag打上存在信号（1）和不存在信号（0）的标签。
用5折验证的方式进行5次分割。


统计5次分割中，每个.wav文件切割成bag后，测试集中存在多少个标签为1和0的bag。

代码用ipynb呈现，代码保存于根目录下。