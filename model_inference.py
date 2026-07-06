import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from typing import Optional
from memory_cal import *
from typing import Optional, Tuple
from SAM_train import parse_args
import tifffile
from SAMamba3D import SAM_Mamba_3D_CoEncoding
from image_slice_view import visualize_slice, visualize_3d_overview
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from Config import Config


class GaussianSlidingWindowInference:

    def __init__(self, model: nn.Module,
                 patch_size: Tuple[int, int, int] = (64, 96, 96),
                 overlap: float = 0.5,
                 sigma_scale: float = 0.125,
                 device: Optional[str] = None):
        self.model      = model
        self.patch_size = patch_size
        self.overlap    = overlap
        self.sigma      = sigma_scale
        self.device     = device

        self._weight = self._gaussian_weight(patch_size, sigma_scale)

    @staticmethod
    def _gaussian_weight(patch_size: Tuple[int, int, int],
                         sigma_scale: float) -> torch.Tensor:
       
        D, H, W = patch_size
        coords = [torch.linspace(-1, 1, s) for s in (D, H, W)]
        grids  = torch.meshgrid(*coords, indexing='ij')
       
        sigma  = 2.0 * sigma_scale
        weight = torch.exp(
            -(grids[0]**2 + grids[1]**2 + grids[2]**2) / (2 * sigma**2)
        )
        return weight  # [D, H, W]

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: [D, H, W]
        Returns:
            pred:   [B, num_classes, D, H, W]
        """
        self.model.eval()
       
        if isinstance(volume, np.ndarray):
            D, H, W = volume.shape
        else:
            D, H, W = volume.shape

        pd, ph, pw = self.patch_size

        
        if self.device is not None:
            dev = torch.device(self.device)
        else:
            try:
                dev = next(self.model.parameters()).device
            except StopIteration:
                dev = torch.device('cpu')

        # 步长（考虑 overlap）
        stride_d = max(1, int(pd * (1 - self.overlap)))
        stride_h = max(1, int(ph * (1 - self.overlap)))
        stride_w = max(1, int(pw * (1 - self.overlap)))

        
        pred_sum = None 
        weight_sum = None
        w_patch = self._weight.unsqueeze(0).cpu().numpy()  # [1,Dp,Hp,Wp]

        def _starts(total, patch, stride):
            starts = list(range(0, total - patch + 1, stride))
            if not starts or starts[-1] + patch < total:
                starts.append(total - patch)
            return starts

        ds_list = _starts(D, pd, stride_d)
        hs_list = _starts(H, ph, stride_h)
        ws_list = _starts(W, pw, stride_w)

        for ds in ds_list:
            for hs in hs_list:
                for ws in ws_list:
                    patch = volume[ds:ds+pd, hs:hs+ph, ws:ws+pw]
                    patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        out_t = self.model.forward_stage1(patch)   # [B, num_classes, pd, ph, pw]
                        out_np = F.softmax(out_t, dim=1)[0].cpu().numpy()  # (C, D, H, W)

                    if pred_sum is None:
                        C = out_np.shape[0]
                        pred_sum = np.zeros((C, D, H, W), dtype=np.float32)
                        weight_sum = np.zeros((1, D, H, W), dtype=np.float32)

                    pred_sum[:, ds:ds+pd, hs:hs+ph, ws:ws+pw] += out_np * w_patch
                    weight_sum[:, ds:ds+pd, hs:hs+ph, ws:ws+pw] += w_patch

        weight_sum = np.maximum(weight_sum, 1e-8)
        pred_avg = pred_sum / weight_sum

        return np.argmax(pred_avg, axis=0).astype(np.uint8)


class InferenceEngine:
    "no gaussian sliding window inference, just for testing"

    def __init__(self, model, config: Config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        self.inference_stats = {}  
        self.infer = GaussianSlidingWindowInference(
                model,
                patch_size   = self.config.patch_size,
                overlap      = 0.5,
                sigma_scale  = 0.125,
            )

    @torch.no_grad()
    def predict_volume(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Returns:
            predictions: (D, H, W) 
        """


        start_time = time.time()
        
        gpu_allocated_before, gpu_reserved_before = get_gpu_memory()
        cpu_memory_before = get_cpu_memory()

        D, H, W = data.shape
        pd, ph, pw = self.config.patch_size
        sd, sh, sw = self.config.stride

     
        pred_sum = np.zeros((D, H, W, self.config.num_classes), dtype=np.float32)
        count_map = np.zeros((D, H, W), dtype=np.float32)

        print(f"Starting sliding window inference...")
        print(f"Volume: {D}x{H}x{W}, Patch: {pd}x{ph}x{pw}, Stride: {sd}x{sh}x{sw}")

        coords = []
        for d in range(0, D, sd):
            d = min(d, D - pd)
            for h in range(0, H, sh):
                h = min(h, H - ph)
                for w in range(0, W, sw):
                    w = min(w, W - pw)
                    coords.append((d, h, w))
        print(f"Total patches to process: {len(coords)}")

        patch_start_time = None
        patch_gpu_allocated = 0
        patch_cpu_memory = 0

  
        for idx, (d, h, w) in enumerate(tqdm(coords, desc="Inference")):
            if idx == 0:
                patch_start_time = time.time()
           
            patch = data[d:d+pd, h:h+ph, w:w+pw]
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
            patch_tensor = patch_tensor.to(self.device)

            seg_pred = self.model.forward_stage1(patch_tensor)
            seg_pred = F.softmax(seg_pred, dim=1)[0].cpu().numpy()  # (C, D, H, W)
            seg_pred = seg_pred.transpose(1, 2, 3, 0)  # (D, H, W, C)

            pred_sum[d:d+pd, h:h+ph, w:w+pw] += seg_pred
            count_map[d:d+pd, h:h+ph, w:w+pw] += 1

            if idx == 0:
                patch_gpu_allocated, _ = get_gpu_memory()
                patch_cpu_memory = get_cpu_memory()
                patch_processing_time = time.time() - patch_start_time

        count_map = np.maximum(count_map, 1)
        pred_avg = pred_sum / count_map[..., np.newaxis]

        
        # Argmax
        predictions = np.argmax(pred_avg, axis=-1).astype(np.uint8)

        end_time = time.time()
        gpu_allocated_after, _ = get_gpu_memory()
        cpu_memory_after = get_cpu_memory()

        total_inference_time = end_time - start_time
        self.inference_stats = {
            'total_time': total_inference_time,
            'time_per_patch': patch_processing_time if 'patch_processing_time' in locals() else 0,
            'total_patches': len(coords),
            'memory': {
                'gpu_allocated_before': gpu_allocated_before,
                'gpu_allocated_after': gpu_allocated_after,
                'gpu_allocated_during_patch': patch_gpu_allocated,
                'cpu_memory_before': cpu_memory_before,
                'cpu_memory_after': cpu_memory_after,
                'cpu_memory_during_patch': patch_cpu_memory
            }
        }
        
        self.inference_stats['memory']['gpu_peak'] = max(
            gpu_allocated_before,
            gpu_allocated_after,
            patch_gpu_allocated
        )
        
        self.inference_stats['memory']['cpu_peak'] = max(
            cpu_memory_before,
            cpu_memory_after,
            patch_cpu_memory
        )

        return predictions

    def predict_and_save(
        self,
        data: np.ndarray,
        output_path: str,
        gt_labels: Optional[np.ndarray] = None
    ):
        print("\n" + "="*60)
        print("Starting Volume Inference")
        print("="*60)

        start_time = time.time()
        predictions = self.predict_volume(data)
        # predictions = self.infer(data)
        end_time = time.time()
        total_inference_time = end_time - start_time


        np.save(output_path, predictions)
        print(f"\nTotal inference time:{format_time(total_inference_time)}")
        print(f"\nPredictions saved to: {output_path}")
        print(f"Prediction shape: {predictions.shape}")


        if gt_labels is not None:
            print("\n" + "-"*60)
            print("Evaluation on Labeled Regions")
            print("-"*60)
            self.evaluate_predictions(predictions, gt_labels)

        return predictions

    def evaluate_predictions(self, predictions: np.ndarray, gt_labels: np.ndarray):
        """评估预测结果"""
       
        gt_classes = gt_labels 
        labeled_mask = np.sum(gt_labels, axis=-1) > 0

        pred_labeled = predictions[labeled_mask]
        gt_labeled = gt_classes[labeled_mask]
    
        print("\nPer-class Metrics (on labeled regions):")
        class_names =  {1: 'Oil', 2: 'Brine', 3: 'Rock', 0: 'Unknown'} 
        overall_correct = 0
        overall_total = 0
        overall_iu = 0

        for cls in [1,2,3]:
            pred_cls = (pred_labeled == cls)
            gt_cls = (gt_labeled == cls)

            tp = np.sum(pred_cls & gt_cls)
            fp = np.sum(pred_cls & ~gt_cls)
            fn = np.sum(~pred_cls & gt_cls)
            tn = np.sum(~pred_cls & ~gt_cls)

            if tp + fp + fn == 0:
                continue

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)

            print(f"\n{class_names[cls]}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  Dice:      {dice:.4f}")
            print(f"  IoU:       {iou:.4f}")

            overall_correct += tp
            overall_total += (tp + fn)
            overall_iu += (tp + fp + fn)

        overall_acc = overall_correct / (overall_total + 1e-8)
        overall_dice = (2 * overall_correct) / (2 * overall_correct + (overall_total - overall_correct) + 1e-8)
        overall_iou = overall_correct / (overall_iu + 1e-8)
        print(f"\nOverall Accuracy: {overall_acc:.4f}")
        print(f"Overall Dice: {overall_dice:.4f}")
        print(f"Overall IoU: {overall_iou:.4f}")
        print(f"Labeled voxels evaluated: {len(gt_labeled)}")


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



def test_model(config: Config, args, checkpoint_path: str,save_filename: str,vis_filename: str, 
               load_datainfo_flag: bool,labels_map: bool):

    print("starting inference...")
    data, labels, data_mean, data_std = load_and_prepare_data(config)
    target_data = np.load("data/Image_SSa.npy").astype(np.float32)
    min_dim = min(data.shape[1], data.shape[2])
      
    if labels is not None:
        if min_dim > len(data):
            test_data= data[:, :min_dim, :min_dim]
            test_labels = labels[:, :min_dim, :min_dim]
        else:
            test_data= data[:min_dim, :min_dim, :min_dim]
            test_labels = labels[:min_dim, :min_dim, :min_dim]
     
    else:
        test_data= data[:, :min_dim, :min_dim]
        test_labels = None

    from Combined_dataloader import percentile_normalization
    test_data = percentile_normalization(test_data,target_data)
    
    # denoise
    # from scipy import ndimage
    # data = ndimage.median_filter(data, size=5)
    # test_data = histogram_matching(test_data,target_data)

    if labels_map and  test_labels is not None:
        test_labels = map_labels(test_labels, config.label_mapping)

    stats_path = os.path.join(config.save_dir, 'data_stats.npy')
    if os.path.exists(stats_path) and load_datainfo_flag:
        stats = np.load(stats_path, allow_pickle=True).item()
        data_mean = stats['global_mean']
        data_std = stats['global_std']
        print(f"Loaded data stats from {stats_path}")

    print("\nLoading model...")
    mamba_config = {
        'in_chans': 1,  # SAM ViT-b hidden dimension
        'depths': [2, 2, 2, 2],
        'dims':[48,96,192,384],
        'drop_path_rate': 0.1,
        'out_indices': [0, 1, 2, 3] 
    }
    model = SAM_Mamba_3D_CoEncoding(

        model_type=args.model_type,
        mamba_config = mamba_config,
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        lora_rank    = 8,
        lora_alpha   = 16.0,

    ).to(config.device)


    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    total_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_size_mb = param_size_bytes / 1024 / 1024
    print("\n Model parameters")
    print(f"Model size (parameters only): {param_size_mb:.2f} MB")
    print(f"Total parameters: {total_params:,}")
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Trained epochs: {checkpoint['epoch']}")
    print(f"Best validation Dice: {checkpoint['best_dice']:.4f}")

    inference_engine = InferenceEngine(model, config, config.device)

    os.makedirs(config.result_dir, exist_ok=True)
    output_path = os.path.join(config.result_dir, save_filename)

    test_data = (test_data - data_mean) / (data_std + 1e-8)
    predictions = inference_engine.predict_and_save(
        data=test_data,
        output_path=output_path,
        gt_labels=test_labels
    )
    

    print("\nGenerating visualizations...")
    mid_slice = 33 
    vis_path = os.path.join(config.result_dir, f'slice_{vis_filename}_{mid_slice}_comparison.png')
    visualize_slice(test_data, predictions, test_labels, mid_slice, vis_path)

    # 3D overview
    overview_path = os.path.join(config.result_dir, f'3Dmamba_{vis_filename}_pred.png')
    overview_label_path = os.path.join(config.result_dir, f'3Dmamba_{vis_filename}_gt.png')
    overview_data_path = os.path.join(config.result_dir, f'3Dmamba_{vis_filename}_raw.png')
    visualize_3d_overview(test_data, predictions, test_labels, overview_path, 
                            overview_label_path,overview_data_path,num_slices=9)

    print("\n" + "="*60)
    print("Testing Completed!")
    print(f"Results saved in: {config.result_dir}")
    print("="*60)

    if hasattr(inference_engine, 'inference_stats') and inference_engine.inference_stats:
        stats = inference_engine.inference_stats
        print("\n" + "="*50)
        print("INFERENCE STATISTICS")
        print("="*50)
        print(f"Total Inference Time: {format_time(stats['total_time'])}")
        print(f"Processed Patches: {stats['total_patches']}")
        print(f"Time per Patch: {stats['time_per_patch']*1000:.2f}ms")
        print("\nMemory Usage:")
        mem = stats['memory']
        print(f"  GPU Memory Before Inference: {mem['gpu_allocated_before']:.1f}MB")
        print(f"  GPU Memory After Inference: {mem['gpu_allocated_after']:.1f}MB")
        print(f"  Peak GPU Memory During Inference: {mem['gpu_peak']:.1f}MB")
        print(f"  CPU Memory Before Inference: {mem['cpu_memory_before']:.1f}MB")
        print(f"  CPU Memory After Inference: {mem['cpu_memory_after']:.1f}MB")
        print(f"  Peak CPU Memory During Inference: {mem['cpu_peak']:.1f}MB")


def load_and_prepare_data(config: Config):
 
    print("Loading data...")
    # 加载原始数据, if .tif..npy
    if config.data_path.endswith('.tif'):
        data = tifffile.imread(config.data_path).astype(np.float32)

        if config.labels_path is not None and config.labels_path.endswith('.tif'):
            labels = tifffile.imread(config.labels_path).astype(np.int64)
        elif config.labels_path is not None and config.data_path.endswith('.np'):
            labels = np.load(config.labels_path).astype(np.int64)
        else:
            labels=None

    else:
        if config.labels_path is not None:
            data = np.load(config.data_path).astype(np.float32)
            labels = np.load(config.labels_path).astype(np.int64)
        else:
            data = np.load(config.data_path).astype(np.float32)
            labels = None

    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
   
    data_mean = np.mean(data)
    data_std = np.std(data)
    print(f"\nData statistics:")
    print(f"  Mean: {data_mean:.4f}")
    print(f"  Std: {data_std:.4f}")
    return data, labels, data_mean, data_std

if __name__ == "__main__":

    config = Config()
    checkpoints = 'patch96_fixv5' 
    config.save_dir = f"/gpfs/home/rzhang2/SAM/checkpoints/run_"+ checkpoints
    config.log_dir = f"/gpfs/home/rzhang2/SAM/logs_edge/run_"+ checkpoints
    config.result_dir = f"/gpfs/home/rzhang2/SAM/results/run_"+ checkpoints
   
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    config.data_path ="/gpfs/home/rzhang2/data/0225/Surfactant-1PV-q500.npy"
    config.labels_path ="/gpfs/home/rzhang2/data/0225/1PV-Surfactant-seg.npy"
    

    import os
    checkpoint_path = os.path.join(config.save_dir, 'best_model.pth')
    save_filename = '1PV_pred.npy'
    vis_filename = '1PV'
    

    config.patch_size = (128,128,128)  # (D, H, W)
    config.stride = tuple(int(p * (1 - 0.7)) for p in config.patch_size)
    config.label_mapping = {
        0:3,
        1:1,
        2:2, # if you want to remap labels, you can do it here
    } 
    
    # inference
    args = parse_args(config)
    test_model(config,args, checkpoint_path,save_filename,vis_filename,
            load_datainfo_flag=True,labels_map= None) #config.label_mapping