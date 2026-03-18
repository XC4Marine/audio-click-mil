# src/dataset.py
# CWD-MIL 项目数据集类定义
# 【当前执行：Phase 0 的任务 2】

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class CWDMILBagDataset(Dataset):
    """
    CWD-MIL 项目专用数据集类
    用于加载预处理的 bags.pkl 文件，支持 train/val/test 分割
    
    【模块：Dataset 正在执行任务 2】
    """
    
    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        split: str = "train",  # train / val / test
        bags_path: str = "data/bags.pkl"
    ):
        """
        初始化数据集
        
        Args:
            config_path: 配置文件路径
            split: 数据分割类型 (train/val/test)
            bags_path: 预处理后的 bags.pkl 文件路径
        """
        # 【当前执行：Phase 0 的任务 2 - 加载配置】
        self.config = self._load_config(config_path)
        self.split = split
        self.bags_path = bags_path
        
        # 加载 bags 数据
        # 【当前执行：Phase 0 的任务 2 - 加载 bags 数据】
        self.bags = self._load_bags(bags_path)
        
        # 按 Ori_file_num 进行 70/15/15 分割
        # 【当前执行：Phase 0 的任务 2 - 数据分割】
        self.samples = self._split_data()
        
        # 特征维度验证
        assert self.config['model']['feat_dim'] == 522, \
            f"feat_dim 必须为 522，当前为 {self.config['model']['feat_dim']}"
    
    def _load_config(self, config_path: str) -> Dict:
        """加载 YAML 配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_bags(self, bags_path: str) -> Dict:
        """加载预处理后的 bags.pkl 文件"""
        with open(bags_path, 'rb') as f:
            bags = pickle.load(f)
        return bags
    
    def _split_data(self) -> List[Dict]:
        """
        按 Ori_file_num 进行 70/15/15 分割
        对应 skill.md: 数据分割规范
        """
        # 获取所有唯一的 Ori_file_num
        unique_files = list(set([bag['Ori_file_num'] for bag in self.bags]))
        
        # 排序确保可复现
        unique_files.sort()
        
        n_files = len(unique_files)
        n_train = int(n_files * 0.70)
        n_val = int(n_files * 0.15)
        # n_test = n_files - n_train - n_val  # 剩余给测试集
        
        # 分割文件列表
        if self.split == "train":
            selected_files = unique_files[:n_train]
        elif self.split == "val":
            selected_files = unique_files[n_train:n_train + n_val]
        elif self.split == "test":
            selected_files = unique_files[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # 筛选属于当前分割的 bags
        samples = [bag for bag in self.bags if bag['Ori_file_num'] in selected_files]
        
        print(f"[Dataset] Split: {self.split}, Files: {len(selected_files)}/{n_files}, "
              f"Samples: {len(samples)}")
        
        return samples
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            Dict 包含:
                - bag_features: (n_instances, feat_dim) 特征矩阵
                - bag_label: (1,) 包级别标签 (0/1)
                - instance_labels: (n_instances,) 实例级别标签 (0/1)
                - bag_id: 包 ID
                - ori_file_num: 原始文件编号
        """
        # 【当前执行：Phase 0 的任务 2 - 获取样本】
        bag = self.samples[idx]
        
        # 提取特征 (n_instances, 522)
        bag_features = torch.FloatTensor(bag['features'])
        
        # 包级别标签 (是否存在 click)
        bag_label = torch.FloatTensor([bag['bag_label']])
        
        # 实例级别标签 (每个 10 秒实例是否有 click)
        instance_labels = torch.FloatTensor(bag['instance_labels'])
        
        return {
            'bag_features': bag_features,           # (n_instances, 522)
            'bag_label': bag_label,                 # (1,)
            'instance_labels': instance_labels,     # (n_instances,)
            'bag_id': bag['bag_id'],
            'ori_file_num': bag['Ori_file_num']
        }


def get_dataloader(
    config_path: str = "configs/default.yaml",
    split: str = "train",
    bags_path: str = "data/bags.pkl",
    batch_size: Optional[int] = None
) -> DataLoader:
    """
    创建 DataLoader
    
    Args:
        config_path: 配置文件路径
        split: 数据分割类型
        bags_path: bags.pkl 路径
        batch_size: 批次大小（若为 None 则从 config 读取）
    
    Returns:
        DataLoader 对象
    """
    # 【当前执行：Phase 0 的任务 2 - 创建 DataLoader】
    dataset = CWDMILBagDataset(
        config_path=config_path,
        split=split,
        bags_path=bags_path
    )
    
    # 从 config 读取 batch_size
    if batch_size is None:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        batch_size = config['train']['batch_size']
    
    # 训练集启用 shuffle
    shuffle = (split == "train")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"[DataLoader] Split: {split}, Batch: {batch_size}, Samples: {len(dataset)}")
    
    return dataloader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数，处理变长实例数量的 bags
    
    【模块：Dataset 正在执行任务 2 - 批处理】
    """
    # 收集所有样本
    bag_features = [item['bag_features'] for item in batch]
    bag_labels = torch.cat([item['bag_label'] for item in batch], dim=0)
    instance_labels = [item['instance_labels'] for item in batch]
    bag_ids = [item['bag_id'] for item in batch]
    ori_file_nums = [item['ori_file_num'] for item in batch]
    
    # 由于每个 bag 的实例数量可能不同，需要 padding 或单独处理
    # 这里返回 list，由 model 内部处理变长序列
    return {
        'bag_features': bag_features,           # List[Tensor], 每个 (n_inst, 522)
        'bag_label': bag_labels,                # (batch_size,)
        'instance_labels': instance_labels,     # List[Tensor]
        'bag_id': bag_ids,
        'ori_file_num': ori_file_nums
    }


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 【当前执行：Phase 0 的任务 2 - 测试验证】
    print("=" * 50)
    print("测试 CWDMILBagDataset 类")
    print("=" * 50)
    
    # 检查 bags.pkl 是否存在
    bags_path = Path("data/bags.pkl")
    if not bags_path.exists():
        print(f"[警告] {bags_path} 不存在，请先运行 scripts/prepare_bags.py")
    else:
        # 测试训练集
        train_loader = get_dataloader(split="train")
        print(f"\n训练集批次数量：{len(train_loader)}")
        
        # 取一个批次验证
        batch = next(iter(train_loader))
        print(f"\n批次结构验证:")
        print(f"  bag_features: {len(batch['bag_features'])} 个 bags")
        print(f"  bag_label: {batch['bag_label'].shape}")
        print(f"  第一个 bag 的实例数: {batch['bag_features'][0].shape[0]}")
        print(f"  特征维度: {batch['bag_features'][0].shape[1]} (应为 522)")
        
        print("\n✅ 数据集类测试通过！")