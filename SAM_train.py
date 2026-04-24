"""
SAM-Mamba 3D训练py
"""

import torch
import argparse
from pathlib import Path
from Combined_dataloader import data_loaders
from Config import Config
# from SAM_Mamba import SAM_Mamba_3D
from mamba_sam_coencoder_fixv4 import SAM_Mamba_3D_CoEncoding
# from mamba_sam_3dcoencoder_v3 import SAM_Mamba_3D_CoEncoding
# from samamba_unet3d import SAMambaUNet3D
# from ablation_sam import build_single
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
            'in_chans': 1,  # SAM ViT-H hidden dimension
            'depths': [2, 2, 2, 2],
            'dims': [48, 96, 192, 384],  # #[32, 64, 128, 256],
            'drop_path_rate': 0.1,
            'out_indices': [0, 1, 2, 3]  # 使用所有层
        }
    
    # 创建模型
    # model = SAM_Mamba_3D(
    #     sam_checkpoint=args.sam_checkpoint,
    #     model_type=args.model_type,
    #     num_classes=args.num_classes,
    #     embed_dim=args.embed_dim,
    #     in_chans=args.in_chans,
    #     out_chans=args.out_chans,
    #     lora_rank=args.lora_rank,
    #     mamba_config = None
    # ).to(device)

    model = SAM_Mamba_3D_CoEncoding(
            model_type=args.model_type,
            mamba_config = mamba_config,
            num_classes=args.num_classes,
            # sam_embed_dim=args.embed_dim,
            in_chans=args.in_chans,
            out_chans=args.out_chans,
            # injection_interval=3,   # 每3层SAM block注入一次
            # layer_scale_init=1e-4,  # 训练初期微弱注入，保护SAM权重
            # window_size=args.window_size,
            lora_rank    = 8,
            lora_alpha   = 16.0,
    ).to(device)

    # model = SAMambaUNet3D(
    #     model_type       = 'vit_b',
    #     mamba_config     = mamba_config,
    #     num_classes      = 4,
    #     feature_size     = 64,
    #     stem_channels    = 32,
    #     hoacm_reduction  = 4,
    #     deep_supervision = False,
    # ).to(device)

    # model.setup_freeze_strategy(args.sam_checkpoint)
    model.set_training_stage(stage='A',sam_checkpoint=args.sam_checkpoint)
    return model

# def create_model_ablation(args,device):
#     """创建消融模型（单阶段）"""
#     model = build_single(device=device,
#                          mode=args.ablation_mode)
#     model.setup_freeze_strategy(args.sam_checkpoint)
#     return model

def main(args,config, data_names,dataset_img_paths, 
                    dataset_label_paths,label_mapping):
    """主训练函数"""
    
    # 设置设备
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
        model = create_model_ablation(args, device)
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
    
    # 执行训练
    if args.stage == 'stage_i' or args.stage == 'both':
        # Stage I Training
        stage_i_checkpoint = Path(config.save_dir) / 'best_model.pth'
        stage_i_best_dice = trainer.train_stage_i(
            checkpoint_path = stage_i_checkpoint,
            num_epochs=args.stage_i_epochs
        )
        # stage_i_best_dice = trainer.fine_tuning(
        #     checkpoint_path = stage_i_checkpoint,
        #     num_epochs=args.stage_i_epochs
        # )
        
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
    # 数据参数
    # parser.add_argument('--train_volume_dir', type=str, default='./data/train_volumes',
    #                    help='Training volumes directory')
    # parser.add_argument('--train_target_dir', type=str, default='./data/train_targets',
    #                    help='Training targets directory')
    # parser.add_argument('--val_volume_dir', type=str, default='./data/val_volumes',
    #                    help='Validation volumes directory')
    # parser.add_argument('--val_target_dir', type=str, default='./data/val_targets',
    #                    help='Validation targets directory')
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
    dataset_img_paths = [
        "/gpfs/home/rzhang2/DPR-125/Image_SSa.npy",
        # "/gpfs/home/rzhang2/DPR-125/Image_SSb.npy",
        "/gpfs/home/rzhang2/data/WF1_image_cor.npy",
        # "/rds/general/user/rzhang2/home/data/Ben_image_crop.npy",
        # "/rds/general/user/rzhang2/home/data/025fw_lowCa_image.npy",
        "/gpfs/home/rzhang2/data/DRP-151/Sample2_image_denoise.npy",
        "/gpfs/home/rzhang2/data/DRP-157/fw85_image.npy",
        # "/gpfs/home/rzhang2/data/DRP-263/fw6_image.npy",
        "/gpfs/home/rzhang2/data/DRP-157/fw30_image.npy",
        # '/gpfs/home/rzhang2/data/DRP-421/LP_image.npy' 
        # "/gpfs/home/rzhang2/data/H2/H2/Filtered_Exp1_Imbibition1.npy"
    

        
    ]
    
    dataset_label_paths = [
        "/gpfs/home/rzhang2/DPR-125/Label_SSa.npy",
        # "/gpfs/home/rzhang2/DPR-125/Label_SSb.npy",
        "/gpfs/home/rzhang2/data/WF1_labels_cor.npy",
        # "/rds/general/user/rzhang2/home/data/Ben_labels_crop.npy",
        # "/rds/general/user/rzhang2/home/data/025fw_lowCa_label_mapping.npy",
        "/gpfs/home/rzhang2/data/DRP-151/Sample2_label_crop.npy",
        "/gpfs/home/rzhang2/data/DRP-157/labels/fw_0.85_curvature_seg.npy", 
        # "/gpfs/home/rzhang2/data/DRP-263/fw0.06_seg.npy",
        "/gpfs/home/rzhang2/data/DRP-157/labels/fw30_label.npy", 
        # '/gpfs/home/rzhang2/data/DRP-421/LP_seg_800.npy'
        #  "/gpfs/home/rzhang2/data/H2/H2/Segmented_Exp1_Imbibition1.npy"
    ]

    labels_mapping = {
        1: 1,  # dataset1: oil
        2: 2,  # dataset1: brine
        3: 3,  # dataset1: rock

        # 3: 2,  
        # 2: 1,  
        # 1: 3,#H2

        # 1: 1,  
        # 2: 2,  
        # 3: 3,

        1: 2,  
        2: 1,  
        3: 3,

        # 1: 2,  
        # 2: 1,  
        # 0: 3,

        # 1: 2,  
        # 2: 1,  
        # 3: 3,

        1: 1,  
        2: 2,  
        3: 3,
        
        1: 1,  
        2: 2,  
        3: 3,

        1: 1,  
        2: 2,  
        3: 3,

        # 0: 1,
        # 1: 2,
        # 2: 3,
        # 3: 0,
        
        
    }
    data_names =  ['SSa','WF1','DRP151-sample2','FW85','FW30'] #,'LP''H2','SSb','DRP58','Ben172'

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
