"""
SAMamba3D train.py
"""

import torch
import argparse
from pathlib import Path
from Combined_dataloader import data_loaders
from Config import Config
from SAMamba3D import SAM_Mamba_3D_CoEncoding
from trainer import SAMMambaTrainer
import numpy as np
import random
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def set_seed(seed=42):
   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_optimizer_scheduler(model, args):
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr = args.learning_rate,
        weight_decay = args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 学习率调度
    # scheduler = None
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_epochs
    )

    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Scheduler: LinearLR")

    return optimizer, scheduler

def create_model(args,device):
    """创建 SAM-Mamba 模型"""
    print("\n Creating SAM-Mamba model...")
    
    # Mamba 配置
    mamba_config = {
            'in_chans': 1,  
            'depths': [2, 2, 2, 2],
            'dims': [48, 96, 192, 384],  
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
    ).to(device)


    model.set_training_stage(stage='A',sam_checkpoint=args.sam_checkpoint)
    return model

def main(args,config, data_names,dataset_img_paths, 
                    dataset_label_paths,label_mapping):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== 第一阶段：Stage I 训练 ====================
    print("\n" + "="*60)
    print("STAGE I: Adapter and Projection Training")
    print("="*60)
    
    # 创建数据加载器
    print("\nLoading data...")
    set_seed()
    
    train_loader, val_loader =  data_loaders(config, data_names,dataset_img_paths, 
                                dataset_label_paths,label_mapping=label_mapping)
    

    if args.ablation_mode is None:
        model = create_model(args, device)
    else:
        print(f"\nCreating ablation model with mode: {args.ablation_mode}")
    optimizer, scheduler = create_optimizer_scheduler(model, args)
    

    # 创建训练器
    trainer = SAMMambaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer= optimizer,
        scheduler=scheduler,
        config=args,
        device=device
    )

    if args.stage == 'stage_i' or args.stage == 'both':
        # Stage I Training
        stage_i_checkpoint = Path(config.save_dir) / 'best_model.pth'
        stage_i_best_dice = trainer.train_stage_i(
            checkpoint_path = stage_i_checkpoint,
            num_epochs=args.stage_i_epochs
        )
      
        
        print("\n" + "="*70)
        print("🎉 Training Completed!")
        print("="*70)
        print(f"Stage I Best Dice: {stage_i_best_dice:.4f}")
    else:
        stage_i_checkpoint = args.resume

    if args.stage == 'stage_ii' or args.stage == 'both':
        # Stage II Training
        stage_ii_best_dice = trainer.train_stage_ii(
            num_epochs=args.stage_ii_epochs,
            stage_i_checkpoint=stage_i_checkpoint if args.stage == 'both' else args.resume
        )

    print("\n" + "="*70)
    print("🎉 Training Completed!")
    print("="*70)
    if args.stage == 'both':
        print(f"Stage I Best Dice: {stage_i_best_dice:.4f}")
        print(f"Stage II Best Dice: {stage_ii_best_dice:.4f}")
    print("="*70)


def parse_args(config):
    """参数解析"""
    parser = argparse.ArgumentParser(
        description='Train SAM-Mamba 3D segmentation model'
    )
    
    # 模型参数
    parser.add_argument('--sam_checkpoint', type=str, default='/gpfs/home/rzhang2/SegMamba/sam_vit_b_01ec64.pth', #,#sam_vit_h_4b8939.pth'sam_vit_l_0b3195.pth'
                       help='SAM checkpoint path')
    parser.add_argument('--model_type', type=str, default='vit_b',
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model type')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of segmentation classes')
    parser.add_argument('--embed_dim', type=int, default=768,
                       help='Embedding dimension')
    parser.add_argument('--in_chans', type=int, default=1,#48
                       help='Input channels')
    parser.add_argument('--out_chans', type=int, default=256,
                       help='Output channels')
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--ablation_mode', type=str, default=None,
                       choices=['M1', 'M2', 'M3', 'M4', None],
                       help='Ablation mode')
    parser.add_argument('--window_size', type=int, default=0,
                       help='attention window size')
   
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=50,
                       help='Save interval')
    
    # 基础配置
    parser.add_argument('--stage', type=str, choices=['stage_i', 'stage_ii', 'both'],
                       default='stage_i', help='Training stage')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--use_amp', type=bool, default=False,
                       help='Use amp')
    
    # 训练参数
    parser.add_argument('--stage_i_epochs', type=int, default=config.num_epochs,
                       help='Number of epochs for Stage I')
    parser.add_argument('--stage_ii_epochs', type=int, default=150,
                       help='Number of epochs for Stage II')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                       help='Number of warmup epochs')
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay,
                       help='Weight decay')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default= config.save_dir,
                       help='Output directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default=config.log_dir,
                       help='Log directory')

    parser.add_argument('--save_name', type=str, default='abl_M3',help='save directory')
    
    return parser.parse_args()


if __name__ == '__main__':

    config = Config()
    print("Patch参数:",config.patch_size)

    
    # 加载training数据
    dataset_img_paths = []
    
    dataset_label_paths = []

    labels_mapping = None
    data_names =  []

    args = parse_args(config)
    timestamp = args.save_name 
    config.save_dir = f"/gpfs/home/rzhang2/SAM/checkpoints/run_"+ timestamp
    config.log_dir = f"/gpfs/home/rzhang2/SAM/logs_edge/run_"+ timestamp
    config.result_dir = f"/gpfs/home/rzhang2/SAM/results/run_"+ timestamp
    args = parse_args(config)
    print("Training configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 创建目录
    import os
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    main(args,config, data_names,dataset_img_paths, 
                    dataset_label_paths,labels_mapping)
