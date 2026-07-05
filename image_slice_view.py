from typing import  Optional
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os
import numpy as np


def visualize_slice(
    data: np.ndarray,
    pred: np.ndarray,
    gt: Optional[np.ndarray],
    slice_idx: int,
    save_path: str
):
    """可视化单个切片"""
    # 提取切片
    data_slice = data[slice_idx]
    pred_slice = pred[slice_idx]

    # 颜色映射
    colors = ['black','red', 'blue', 'green', 'yellow']
    cmap = ListedColormap(colors)

    if gt is not None:
        # 转换GT: (H, W, 3) -> (H, W)
        gt_slice = gt[slice_idx] #np.argmax(gt[slice_idx], axis=-1)
        # 未标注区域设为-1
        gt_slice[np.sum(gt_slice, axis=-1) == 0] = -1

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(data_slice, cmap='gray')
        axes[0].set_title('Original CT')
        axes[0].axis('off')

        axes[1].imshow(data_slice, cmap='gray')
        im1 = axes[1].imshow(pred_slice, alpha=0.9, cmap=cmap, vmin=0, vmax=4)
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        axes[2].imshow(data_slice, cmap='gray')
        im2 = axes[2].imshow(gt_slice, alpha=0.9, cmap=cmap, vmin=0, vmax=4)
        axes[2].set_title('Base Case')
        axes[2].axis('off')


        legend_elements = [
            # Patch(facecolor='black', label='Rock'),
            Patch(facecolor='red', label='Brine'),
            Patch(facecolor='blue', label='Oil'),
            Patch(facecolor='green', label='Grain'),
        
        ]
        # fig.legend(handles=legend_elements, loc='lower center', ncol=5,
        #           bbox_to_anchor=(0.5, -0.05))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(data_slice, cmap='gray')
        axes[0].set_title('Original CT')
        axes[0].axis('off')

        axes[1].imshow(data_slice, cmap='gray')
        axes[1].imshow(pred_slice, alpha=0.9, cmap=cmap, vmin=0, vmax=4)
        axes[1].set_title('Prediction')
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=330, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {save_path}")

def visualize_3d_overview(
    data: np.ndarray,
    pred: np.ndarray,
    gt: Optional[np.ndarray],
    save_path: str,
    save_label_path:Optional[str],
    save_data_path:Optional[str],
    num_slices: int = 9
):
    """可视化多个切片overview"""
    D = pred.shape[0]
    indices = np.linspace(0,D-20, num_slices, dtype=int)

    colors = ['black', 'red', 'blue', 'green', 'yellow']
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        axes[i].imshow(pred[idx], cmap=cmap, vmin=0, vmax=4)
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=330, bbox_inches='tight')
    plt.close()
    print(f"3D pred seg overview saved: {save_path}")
    
    if gt is not None:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            axes[i].imshow(gt[idx], cmap=cmap, vmin=0, vmax=4)
            axes[i].set_title(f'Slice {idx}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(save_label_path, dpi=330, bbox_inches='tight')
        plt.close()
        print(f"3D base case overview saved: {save_label_path}")
    if data is not None:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            axes[i].imshow(data[idx], cmap='gray')
            axes[i].set_title(f'Slice {idx}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(save_data_path, dpi=330, bbox_inches='tight')
        plt.close()
        print(f"3D raw data overview saved: {save_data_path}")



checks = 'patch96_fixv5' 
# update save_dir, log_dir, result_dir with timestamp
save_dir = f"/gpfs/home/rzhang2/SAM/checkpoints/run_"+ checks
log_dir = f"/gpfs/home/rzhang2/SAM/logs_edge/run_"+ checks
result_dir = f"/gpfs/home/rzhang2/SAM/results/run_"+ checks

data_path = "" # add your data path here
labels_path = ""


# import tifffile
# data = tifffile.imread(data_path).astype(np.int64)
# gt =  tifffile.imread(labels_path).astype(np.int64)

data = np.load(data_path)
gt = np.load(labels_path)

slice = "123"  # specify the slice index you want to visualize
pred = np.load(os.path.join(result_dir, 'fw0.24_pred.npy'))
save_path = os.path.join(result_dir, "fw0.24_slice"+slice+".png")

def map_labels(labels, label_mapping):
    """
    标签映射
    """
    mapped_labels = np.zeros_like(labels)
    for old, new in label_mapping.items():
        mapped_labels[labels == old] = new
    return mapped_labels

import numpy as np
label_mapping = {
            1: 2,  
            0: 1,  
            2: 3,
        } 
# gt = map_labels(gt,label_mapping)
visualize_slice(data,pred,gt,int(slice),save_path)

# vis_filename = 'fw24'
# overview_path = os.path.join(result_dir, f'3Dmamba_{vis_filename}_overview_96.png')
# overview_label_path = os.path.join(result_dir, f'3Dmamba_{vis_filename}_gt_overview_96.png')
# visualize_3d_overview(pred, gt, overview_path, overview_label_path,num_slices=9)