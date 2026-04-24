import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from pathlib import Path
import time
from early_stopping import EarlyStopping
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from Compoundloss import RockCoreLoss,EnhancedLoss

class SAMMambaTrainer:
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,
        device='cuda',
        num_class = 4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.num_class = num_class
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.losses = RockCoreLoss()  #EnhancedLoss()
        
        # Mixed Precision Training
        self.use_amp = getattr(config, 'use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 日志和检查点
        self._setup_logger()
        self.best_dice = 0.0
        self.current_stage = None
        
        # 训练统计
        self.train_metrics = {
            'loss': [], 'dice_loss': [], 'ce_loss': []
        }
        self.val_metrics = {
            'loss': [], 'dice': []
        }

        self.early_stopping = EarlyStopping(
            patience=25,  
            min_delta=0.001,  # 最小提升阈值
            restore_best_weights=False  # 恢复最佳权重
        )
    
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    
    def dice_loss(self, pred, target, smooth=1e-8):
        """Dice损失(边缘加权)"""

        pred = F.softmax(pred, dim=1)
        # 确保标签在有效范围内 
        target = torch.clamp(target.long(), min=0, max=self.num_class - 1)
        
        # 使用正确的num_classes参数，与实际类别数匹配
        target_onehot = F.one_hot(target, num_classes=self.num_class).permute(0, 4, 1, 2, 3).float()

        # 计算每个类别的Dice
        dice_loss = 0
        for c in range(pred.shape[1]):
            pred_c = pred[:, c]
            target_c = target_onehot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice_coeff = (2. * intersection + smooth) / (union + smooth)
            dice_loss = dice_loss + (1 - dice_coeff)
        
        return dice_loss / pred.shape[1]
        
    def train_stage_i(self, checkpoint_path=None, num_epochs=200):
        """
        Stage I: 训练 Adapters 和 Pseudo Mask Generator
        
        目标: 让 Adapters 学会提取 polyp-specific features
        """
        print("Starting Training...")
        print("="*70)
        
        self.current_stage = 'stage_i'
        
        # 冻结 SAM，只训练 Adapters
        # self._setup_stage_i(checkpoint_path)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            self._load_checkpoint(checkpoint_path)   
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            if epoch > self.config.warmup_epochs and epoch <= self.config.warmup_epochs+100:
                self.model.set_training_stage('B')
                # self.optimizer = torch.optim.AdamW(
                #     [
                #     {"params": self.model.co_encoder.sam_neck.parameters(), "lr": self.config.learning_rate * 0.1},
                #     ],
        
                #     weight_decay=self.config.weight_decay
                # )
            if epoch > self.config.warmup_epochs+100:
                self.model.set_training_stage('C')
            train_metrics = self._train_epoch_stage_i(epoch, num_epochs)
            # 验证
            val_metrics = self._validate_stage_i(epoch)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录指标
            self._log_epoch(epoch, train_metrics, val_metrics, 
                          time.time() - epoch_start_time)
            
            # 保存检查点
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self._save_checkpoint(epoch, 'best_model.pth')
                print(f"✅ Best model saved! Dice: {self.best_dice:.4f}")
            print(f"{'='*70}")
            
            # 定期保存
            if epoch % getattr(self.config, 'save_interval', 30) == 0:
                self._save_checkpoint(epoch, f'stage_i_epoch_{epoch}.pth')
            
            # 检查是否需要早停
            if self.early_stopping(val_metrics['dice'], self.model, epoch):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best validation Dice was {self.early_stopping.best_score:.4f} at epoch {self.early_stopping.best_epoch}")
                # 恢复最佳权重
                # self.early_stopping.restore_best_weights_if_needed(self.model)
                # self.logger.info("Restored best model weights")
                # print("Restored best model weights")
                # 保存最后的权重
                self._save_checkpoint(epoch, 'stage_i_final.pth')
                break
        
        print(f"\n✅ Stage I Training Completed! Best Dice: {self.best_dice:.4f}")
        return self.best_dice
    
    
    def _train_epoch_stage_i(self, epoch, total_epochs):
        """Stage I 的单个 epoch 训练"""
        self.model.train()
        
        metrics = {
            'loss': 0, 'dice_loss': 0, 'ce_loss': 0,'dice':0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Stage I Epoch {epoch}/{total_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images= batch['data'].to(self.device)  # [B, 1, D, H, W]
            masks = batch['label'].to(self.device) # [B, D, H, W]
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp,device_type='cuda'):
                outputs = self.model.forward_stage1(
                    images, 
                    gt_masks=masks, 
                    training=True
                )
                
                final_output = outputs['final_output']
                # consist_losses = outputs['consistency_loss']
                # 对齐output and masks的维度
                if final_output.shape[2:] != masks.shape[1:]:
                    final_output = F.interpolate(final_output, size=masks.shape[1:], mode='trilinear', align_corners=False)
                # print(f"第一阶段最终输出维度 {final_output.shape}")[2, 1, 128, 128]
                
                # 计算 Loss: L_Stage-I = L_D(M, O_Enc^Up)
                # loss_dice = self.dice_loss(final_output, masks)
                # loss_ce = self.ce_loss(final_output, masks)
                # total_loss = loss_dice + loss_ce #+ consist_losses
                total_loss, metrices = self.losses(final_output, masks)

                # 深度监督损失：中间层输出的损失
                # loss_ds = 0
                # ds_weights = [0.5, 0.7, 0.85, 1.0]  # 越深的层权重越大
                
                # for i, (inter_output, weight) in enumerate(zip(
                #     intermediate_outputs, 
                #     ds_weights
                # )):
                #     loss_dice_inter = self.dice_loss(inter_output, masks)
                #     loss_bce_inter = self.bce_loss(inter_output, masks)
                #     loss_ds += weight * (loss_dice_inter + loss_bce_inter)
                
                # # 归一化深度监督损失
                # if len(intermediate_outputs) > 0:
                #     loss_ds = loss_ds / len(intermediate_outputs)
                # total_loss = main_loss + 0.4 * loss_ds 
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                pred_label = torch.argmax(final_output, dim=1)
                dice = self.calculate_dice(pred_label, masks, 4)
            
            # 累计指标
            metrics['loss'] += total_loss.item()
            metrics['dice_loss'] += metrices['dice'].item()
            metrics['ce_loss'] += metrices['interior'].item()
            metrics['dice'] += dice
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Dice': f"{dice:.4f}",
                # 'DeepSuper': f"{loss_ds.item() if isinstance(loss_ds, torch.Tensor) else loss_ds:.4f}",
            })
        
        # 平均指标
        num_batches = len(self.train_loader)
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    @torch.no_grad()
    def _validate_stage_i(self, epoch):
        """Stage I 验证"""
        self.model.eval()
        
        metrics = {'loss': 0, 'dice': 0}
        
        for batch in tqdm(self.val_loader, desc=f"Validation"):
            images =  batch['data'].to(self.device)
            masks = batch['label'].to(self.device)
            
            final_output = self.model.forward_stage1(
                images, 
                gt_masks=None, 
                training=False
            )
            
            # loss_dice = self.dice_loss(final_output, masks)
            # loss_bce = self.ce_loss(final_output, masks)
            # total_loss = loss_dice + loss_bce
            total_loss, metrices = self.losses(final_output, masks)
            
            # 计算指标
            pred_label = torch.argmax(final_output, dim=1)
            dice = self.calculate_dice(pred_label, masks, 4)
            
            metrics['loss'] += total_loss.item()
            metrics['dice'] += dice

        # 平均
        num_batches = len(self.val_loader)
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    

    def calculate_dice(self, pred, target, num_classes):
        """计算Dice (仅在mask区域)"""
        dice_sum = 0
        valid_classes = 0
        smooth = 1e-5

        for c in range(1,num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > 0:
                dice_sum += (2.0 * intersection + smooth) / (union + smooth)
                valid_classes += 1

        return dice_sum / max(valid_classes, 1)
    
    def _setup_stage_i(self,checkpoint_path):
        """配置 Stage I 训练参数"""
        
        print("\n🔧 Setting up Stage I...")
        # 加载检查点
        if checkpoint_path and os.path.isfile(checkpoint_path):
            self._load_checkpoint(checkpoint_path)   
        else:
            sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
            # sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
            mask_generator = SamAutomaticMaskGenerator(sam)
            static_imageencoder = mask_generator.predictor.model.image_encoder.state_dict()
            self.model.sam_img_encoder.load_state_dict(static_imageencoder, strict=False)
            del sam, mask_generator
            torch.cuda.empty_cache()
            print("✓ Loaded SAM (ViT-B) image encoder weights")

        # 冻结 SAM Encoder
        for param in self.model.sam_img_encoder.parameters():
            param.requires_grad = False
        
        for i in self.model.sam_img_encoder.neck_3d:
            for p in i.parameters():
                p.requires_grad = True
        for i in self.model.sam_img_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            for p in i.adapter.parameters():
                p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
        
        # # 冻结 Prompt Encoder
        # for param in self.model.prompt_encoder.parameters():
        #     param.requires_grad = False
        
        # # 冻结 Mask Decoder
        # for param in self.model.mask_decoder.parameters():
        #     param.requires_grad = False
        
        # # 解冻 Adapters
        # for adapter in self.model.progressive_adapters:
        #     for param in adapter.parameters():
        #         param.requires_grad = True
        # for param in self.model.primary_adapter.parameters():
        #     param.requires_grad = True
        
        # 解冻 Pseudo Mask Generator
        # for param in self.model.pseudo_mask_generator.parameters():
        #     param.requires_grad = True
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"  Frozen parameters: {(total_params - trainable_params) / 1e6:.2f}M")
        print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    
    def _log_epoch(self, epoch, train_metrics, val_metrics, epoch_time):
        """记录 epoch 信息"""
        lr = self.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch} | Time: {epoch_time:.2f}s | LR: {lr:.2e}")
        
        print(f"Train - Loss: {train_metrics['loss']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f}  "
             )
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f}"
             )
        # print(f"{'='*70}")
        self.logger.info(
            f" Total loss: {train_metrics['loss']:.4f} | Dice: {train_metrics['dice']:.4f} | \
            Val_loss: {val_metrics['loss']:.4f}| Val_Dice: {val_metrics['dice']:.4f}")
    
    def _save_checkpoint(self, epoch, filename):
        """保存检查点"""
        checkpoint_path = Path(self.config.save_dir) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'stage': self.current_stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_dice': self.best_dice,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # self.best_dice = checkpoint.get('best_dice', 0.0)
        start_epoch = checkpoint.get('epoch', 0) 
       
        print(f"✅ Checkpoint loaded from: {checkpoint_path} at epoch {start_epoch}")
        print(f"   Previous best Dice: {checkpoint.get('best_dice', 0):.4f}")
    
    def fine_tuning(self, checkpoint_path,num_epochs):
        
        print("="*60)
        print("Fine-tuning...")

        # 加载检查点
        if checkpoint_path and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_dice = checkpoint.get('best_dice', 0.0)
            start_epoch = checkpoint.get('epoch', 0) + 1
            self.logger.info(f"Resumed from {checkpoint_path} at epoch {start_epoch}")
            print(f"Resumed from {checkpoint_path} at epoch {start_epoch}")

        # ===============================
        # Fine-tuning parameters
        # ===============================
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        for param in self.model.co_encoder.sam_neck.parameters():
            param.requires_grad = True
        
        self.optimizer = torch.optim.AdamW(
            [
            # {"params": self.model.decoder1.parameters(), "lr": self.config.learning_rate * 0.1},
            # {"params": self.model.decoder2.parameters(), "lr": self.config.learning_rate * 0.1},
            {"params": self.model.decoder.parameters(), "lr": self.config.learning_rate * 0.1},
            {"params": self.model.co_encoder.sam_neck.parameters(), "lr": self.config.learning_rate * 0.05 },
        ],
            weight_decay=self.config.weight_decay
        )


        for epoch in range(1,num_epochs + 1):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch_stage_i(epoch, num_epochs)
            # 验证
            val_metrics = self._validate_stage_i(epoch)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录指标
            self._log_epoch(epoch, train_metrics, val_metrics, 
                          time.time() - epoch_start_time)
            # 保存检查点
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self._save_checkpoint(epoch, 'best_model.pth')
                print(f"✅ Best model saved! Dice: {self.best_dice:.4f}")
            print(f"{'='*70}")
            
            # 定期保存
            if epoch % getattr(self.config, 'save_interval', 30) == 0:
                self._save_checkpoint(epoch, f'stage_i_epoch_{epoch}.pth')
            
            # 检查是否需要早停
            if self.early_stopping(val_metrics['dice'], self.model, epoch):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best validation Dice was {self.early_stopping.best_score:.4f} at epoch {self.early_stopping.best_epoch}")
                # 恢复最佳权重
                # self.early_stopping.restore_best_weights_if_needed(self.model)
                # self.logger.info("Restored best model weights")
                # print("Restored best model weights")
                # 保存最后的权重
                self._save_checkpoint(epoch, 'stage_i_final.pth')
                break
        
        print(f"\n✅ Stage I Training Completed! Best Dice: {self.best_dice:.4f}")
        return self.best_dice