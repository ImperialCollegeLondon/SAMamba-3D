from scipy.ndimage import sobel, binary_dilation
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import random
from typing import List, Tuple, Dict, Optional

class EdgeDetector:
    """3D边缘检测"""

    @staticmethod
    def extract_edges_3d(data: np.ndarray, thickness: int = 2) -> np.ndarray:
        """提取3D图像边缘"""
        # 计算梯度
        grad_x = np.abs(sobel(data.astype(np.float32), axis=0))
        grad_y = np.abs(sobel(data.astype(np.float32), axis=1))
        grad_z = np.abs(sobel(data.astype(np.float32), axis=2))

        # 梯度幅值
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        # 使用自适应阈值提取边缘
        threshold = np.mean(grad_magnitude) + np.std(grad_magnitude)
        edge_mask = (grad_magnitude > threshold).astype(np.float32)

        # 膨胀
        if thickness > 1:
            edge_mask = binary_dilation(edge_mask, iterations=thickness-1).astype(np.float32)

        return edge_mask



class Unified3DPatchDataset(Dataset):
    def __init__(
        self,
        datas: List[np.ndarray],
        labels: List[np.ndarray],
        patch_size: Tuple[int, int, int],
        num_patches: int,
        config: any,
        global_stats: Optional[Tuple[float, float]] = None, # (mean, std)
        augmentation: bool = True,
        mode: str = 'train',
        valid_threshold: float = 0.1
    ):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.config = config
        self.augmentation = augmentation
        self.mode = mode    
        self.labels = labels
        self.all_data = datas

        # 计算或获取全局统计量 (Global Z-score Stats)
        # 统计量是基于已经 [0, 1] 对齐后的数据计算的
        if global_stats is None:
            if len(self.all_data) < 2:
                all_aligned_data = np.array(self.all_data).ravel()
            else:
                all_aligned_data = np.concatenate([d.ravel() for d in self.all_data])
            self.global_mean = np.mean(all_aligned_data)
            self.global_std = np.std(all_aligned_data)
            del all_aligned_data # 释放内存
        else:
            self.global_mean, self.global_std = global_stats

        # 扫描坐标池
        self.coords_pool = self._build_coords_pool(valid_threshold)
        print(f"[{mode.upper()}] Dataset initialized. Mean: {self.global_mean:.4f}, Std: {self.global_std:.4f}")
        print(f"[{mode.upper()}] Total patches available: {len(self.coords_pool)}")

    def _build_coords_pool(self, threshold: float) -> List[Tuple[int, int, int, int]]:
        stride = 32 #if self.mode == 'train' else 16
        coords = []
        pd, ph, pw = self.patch_size

        for idx, label in enumerate(self.labels):
            D, H, W = label.shape
            for d in range(0, D - pd + 1, stride):
                for h in range(0, H - ph + 1, stride):
                    for w in range(0, W - pw + 1, stride):
                        if self.mode == 'train':
                            # 训练集：只保留有标注的区域
                            patch_label = label[d:d+pd, h:h+ph, w:w+pw]
                            if (np.sum(patch_label != 0) / patch_label.size) >= threshold:
                                coords.append((idx, d, h, w))
                        else:
                            # 验证/测试集：全量收集
                            coords.append((idx, d, h, w))
        return coords if coords else [(0, 0, 0, 0)]

    def _augment(self, data: np.ndarray, label: np.ndarray):
        # 旋转与翻转
        for axis in range(3):
            if random.random() > 0.5:
                data = np.flip(data, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        if random.random() > 0.5:
            k = random.randint(1, 3)
            data = np.rot90(data, k=k, axes=(1, 2)).copy()
            label = np.rot90(label, k=k, axes=(1, 2)).copy()
        
        # 亮度
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            data = data * factor

        # 噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02, data.shape)
            data = data + noise
    
        return data, label

    def __len__(self):
        return self.num_patches

    def __getitem__(self, _):
        # 随机选取坐标
        v_idx, d, h, w = random.choice(self.coords_pool)
        pd, ph, pw = self.patch_size
        
        # 训练模式下的随机微调偏移
        if self.mode == 'train':
            D, H, W = self.all_data[v_idx].shape
            d = np.clip(d + random.randint(-8, 8), 0, D - pd)
            h = np.clip(h + random.randint(-8, 8), 0, H - ph)
            w = np.clip(w + random.randint(-8, 8), 0, W - pw)

        # 获取数据
        data_patch = self.all_data[v_idx][d:d+pd, h:h+ph, w:w+pw].copy()
        label_patch = self.labels[v_idx][d:d+pd, h:h+ph, w:w+pw].copy()

        # 增强
        if self.mode == 'train' and self.augmentation:
            data_patch, label_patch = self._augment(data_patch, label_patch)

        # 全局 Z-score 归一化
        data_patch = (data_patch - self.global_mean) / (self.global_std + 1e-8)

        edge_mask = EdgeDetector.extract_edges_3d(label_patch, thickness=2)

        return {
            'data': torch.from_numpy(data_patch).float().unsqueeze(0),
            'label': torch.from_numpy(label_patch).long(),
            'known_mask': torch.from_numpy((label_patch != 0).astype(np.float32)).float(),
            'edge': torch.from_numpy(edge_mask).float()
        }


def get_train_val_test(data_names,datas, labels, patch_size, num_patches_list, config,global_stats=None):
    """
    划分数据并返回三个 Dataset 实例
    num_patches_list: [train_num, val_num, test_num]
    """
  
    n = len(datas)
    indices = list(range(n))
    random.shuffle(indices)
    
    data_splits = []
    label_splits = []

    for i, (dataname,dataset, labels) in enumerate(zip(data_names,datas, labels)):
        total_slices = dataset.shape[0]
        if total_slices < 800:
            train_ratio, val_ratio = 0.8, 0.2
        elif total_slices >= 3000:
            train_ratio, val_ratio = 0.3, 0.1
        else:
            train_ratio, val_ratio = 0.5, 0.1

        split_a = int(total_slices * train_ratio)
        split_b = int(total_slices * (train_ratio + val_ratio))
        # 图片修剪成正方形
        min_dim = min(dataset.shape[1], dataset.shape[2])
        dataset = dataset[:, :min_dim, :min_dim]
        labels = labels[:, :min_dim, :min_dim]

        if dataname == 'SSa' or dataname == 'SSb':
            data_splits.append((
                dataset[:split_a, :, :],
                dataset[split_a:split_b, :, :],
                dataset[split_b:, :, :]
            ))
            label_splits.append((
                labels[:split_a, :, :],
                labels[split_a:split_b, :, :],
                labels[split_b:, :, :]
            ))
        elif dataname=='FW85' or dataname =='FW30' or dataname =='FW24':
            data_splits.append((
                    dataset[100:split_a, :, :],
                    dataset[split_a:split_b, :, :],
                    dataset[split_b:, :, :]
                ))
            label_splits.append((
                    labels[100:split_a, :, :],
                    labels[split_a:split_b, :, :],
                    labels[split_b:, :, :]
                ))
     
        else:
            data_splits.append((
                dataset[50:split_a, :, :],
                dataset[split_a:split_b, :, :],
                dataset[split_b:, :, :]
            ))
            label_splits.append((
                labels[50:split_a, :, :],
                labels[split_a:split_b, :, :],
                labels[split_b:, :, :]
            ))
        print(f"Dataset {dataname}: train/val/test slices: {data_splits[-1][0].shape[0]}/{data_splits[-1][1].shape[0]}/{data_splits[-1][2].shape[0]}")


    # 1. 创建训练集并计算全局统计量
    train_ds = Unified3DPatchDataset(
        [data_splits[i][0] for i in range(len(data_splits))],
        [label_splits[i][0] for i in range(len(label_splits))],
        patch_size, num_patches_list[0], config,global_stats, mode='train'
    )
    stats = (train_ds.global_mean, train_ds.global_std)
    np.save(os.path.join(config.save_dir, 'data_stats.npy'), {'global_mean': train_ds.global_mean, 'global_std': train_ds.global_std})

    # 2. 创建验证集和测试集 (继承训练集的统计量)
    val_ds = Unified3DPatchDataset(
            [data_splits[i][1] for i in range(len(data_splits))],
            [label_splits[i][1] for i in range(len(label_splits))],
            patch_size, num_patches_list[1], config, 
            global_stats=stats, augmentation=False, mode='val'
    )
    
    # test_ds = Unified3DPatchDataset(
    #         [data_splits[i][2] for i in range(len(data_splits))],
    #     [label_splits[i][2] for i in range(len(label_splits))],
    #     patch_size, num_patches_list[2], config, 
    #     global_stats=stats, augmentation=False, mode='test'
    # )

    return train_ds, val_ds #, test_ds

from torch.utils.data import Dataset, DataLoader
def map_labels(labels, label_mapping):
        """
        标签映射
        """
        if label_mapping is None:
            return labels
        else:
            mapped_labels = np.zeros_like(labels)
            for old, new in label_mapping.items():
                mapped_labels[labels == old] = new
            return mapped_labels

def percentile_normalization(source: np.ndarray, target: np.ndarray,
                                 percentiles: Tuple[int, int] = (1, 99)) -> np.ndarray:
        """
        百分位归一化 - 基于百分位数对齐图像
        
        Args:
            source: 源域图像
            target: 目标域图像
            percentiles: 百分位数范围
        
        Returns:
            对齐后的图像
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        
        # 计算百分位数
        src_low, src_high = np.percentile(source, percentiles)
        tgt_low, tgt_high = np.percentile(target, percentiles)
        
        # 线性映射
        aligned = (source - src_low) / (src_high - src_low + 1e-8)
        aligned = aligned * (tgt_high - tgt_low) + tgt_low
        aligned = np.clip(aligned, tgt_low, tgt_high)
        
        return aligned.astype(np.float32)
def data_loaders(config, data_names, data_paths, label_paths, label_mapping=None):
    """
    数据加载器生成函数
    """
    # 加载所有数据集
    data_all = [np.load(dp).astype(np.float32) for dp in data_paths]
    labels_all = [np.load(lp).astype(np.int32) for lp in label_paths]
    labels_all = [map_labels(lbl,label_mapping) for lbl in labels_all]

    # data_mean = np.mean([np.mean(data) for data in data_all])
    # data_std = np.std([np.std(data) for data in data_all])
    # global_stats = (data_mean, data_std)
   
    # 以dataset1为target进行直方图/百分比归一匹配
    for i in range(len(data_all)):
        if i !=0:
            data_all[i] = percentile_normalization(data_all[i],data_all[0])
            # np.save(data_names[i]+f"_transfer_to_SSa.npy",data_all[i])

    train_set, val_set = get_train_val_test(
    data_names= data_names,
    datas=data_all, 
    labels=labels_all, 
    patch_size = config.patch_size, 
    num_patches_list = [config.train_patches,config.val_patches,config.val_patches] ,
    config=config,
    global_stats = None
)
    
    # DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches per epoch (len_trainloader): {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader