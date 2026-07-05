import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """解决极度不平衡问题，专注于难分样本"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重 [C]
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        pred: [B, C, D, H, W]
        target: [B, D, H, W]
        """
        # 计算softmax概率
        p = F.softmax(pred, dim=1)
        
        # 获取目标类别的概率
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Focal loss公式
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        # 类别权重
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target.view(-1)).view_as(target)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# 边界精度约束
class BoundaryLoss(nn.Module):
    """专注于边界区域的损失"""
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, pred, target):
        """
        使用距离变换强调边界
        """
        # 计算边界
        boundary = self.compute_boundary(target)
        
        # 计算距离变换
        dist_map = self.compute_distance_transform(target)
        
        # Softmax预测
        pred_soft = F.softmax(pred, dim=1)
        
        # 在边界附近加权
        weight = torch.exp(-dist_map / self.theta)
        
        # 计算加权损失
        loss = 0
        for c in range(pred.shape[1]):
            target_c = (target == c).float()
            loss += (weight * (pred_soft[:, c] - target_c) ** 2).mean()
        
        return loss
    
    def compute_boundary(self, target):
        """计算3D边界"""
        # 计算梯度
        grad_d = torch.abs(target[:, 1:, :, :] - target[:, :-1, :, :])
        grad_h = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        grad_w = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        boundary = torch.zeros_like(target, dtype=torch.float)
        boundary[:, :-1, :, :] += (grad_d > 0).float()
        boundary[:, :, :-1, :] += (grad_h > 0).float()
        boundary[:, :, :, :-1] += (grad_w > 0).float()
        
        return (boundary > 0).float()
    
    def compute_distance_transform(self, target):
        """计算到边界的距离"""
        from scipy.ndimage import distance_transform_edt
        
        boundary = self.compute_boundary(target)
        dist_maps = []
        
        for b in range(target.shape[0]):
            dist = distance_transform_edt(1 - boundary[b].cpu().numpy())
            dist_maps.append(torch.from_numpy(dist))
        
        return torch.stack(dist_maps).to(target.device)
    
class EdgeAwareLoss(nn.Module):
    """结合图像梯度的边界感知损失"""
    def __init__(self, edge_weight=2.0):
        super().__init__()
        self.edge_weight = edge_weight
    
    def forward(self, pred, target, image):
        """
        pred: [B, C, D, H, W]
        target: [B, D, H, W]
        image: [B, 1, D, H, W]
        """
        # 基础CE损失
        ce_loss = F.cross_entropy(pred, target)
        
        # 计算图像边缘（高梯度区域）
        image_edges = self.compute_image_gradient(image)
        
        # 计算预测边缘
        pred_soft = F.softmax(pred, dim=1)
        pred_edges = self.compute_prediction_gradient(pred_soft)
        
        # 边缘对齐损失：预测边缘应该在图像边缘
        edge_loss = F.mse_loss(pred_edges, image_edges)
        
        return ce_loss + self.edge_weight * edge_loss
    
    def compute_image_gradient(self, image):
        """计算3D图像梯度幅值"""
        grad_d = torch.abs(image[:, :, 1:, :, :] - image[:, :, :-1, :, :])
        grad_h = torch.abs(image[:, :, :, 1:, :] - image[:, :, :, :-1, :])
        grad_w = torch.abs(image[:, :, :, :, 1:] - image[:, :, :, :, :-1])
        
        # Padding
        grad_d = F.pad(grad_d, (0, 0, 0, 0, 0, 1))
        grad_h = F.pad(grad_h, (0, 0, 0, 1, 0, 0))
        grad_w = F.pad(grad_w, (0, 1, 0, 0, 0, 0))
        
        gradient_magnitude = torch.sqrt(grad_d**2 + grad_h**2 + grad_w**2 + 1e-10)
        return gradient_magnitude
    
    def compute_prediction_gradient(self, pred_soft):
        """计算预测的梯度"""
        # 计算最大概率的变化
        max_prob, _ = pred_soft.max(dim=1, keepdim=True)
        
        grad_d = torch.abs(max_prob[:, :, 1:, :, :] - max_prob[:, :, :-1, :, :])
        grad_h = torch.abs(max_prob[:, :, :, 1:, :] - max_prob[:, :, :, :-1, :])
        grad_w = torch.abs(max_prob[:, :, :, :, 1:] - max_prob[:, :, :, :, :-1])
        
        grad_d = F.pad(grad_d, (0, 0, 0, 0, 0, 1))
        grad_h = F.pad(grad_h, (0, 0, 0, 1, 0, 0))
        grad_w = F.pad(grad_w, (0, 1, 0, 0, 0, 0))
        
        return torch.sqrt(grad_d**2 + grad_h**2 + grad_w**2 + 1e-10)

# 区域约束
class TverskyLoss(nn.Module):
    """Tversky Loss - Dice的泛化，可以调节FP和FN的权重"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # FP权重
        self.beta = beta    # FN权重
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target = torch.clamp(target.long(),1, pred.shape[1])
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        tversky_loss = 0
        for c in range(pred.shape[1]):
            pred_c = pred_soft[:, c]
            target_c = target_one_hot[:, c]
            
            TP = (pred_c * target_c).sum()
            FP = (pred_c * (1 - target_c)).sum()
            FN = ((1 - pred_c) * target_c).sum()
            
            tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
            tversky_loss += (1 - tversky)
        
        return tversky_loss / pred.shape[1]

class EnhancedLoss(nn.Module):
    """增强组合损失"""

    def __init__(self, num_classes: int = 4,
                edge_weight: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        self.smooth = 1e-8
        self.ce_loss = nn.CrossEntropyLoss()

        self.focal = FocalLoss()
        self.boundary = EdgeAwareLoss()
        self.tversky = TverskyLoss(beta=0.6)

    def dice_loss(self, pred, target):
        """Dice损失(边缘加权)"""

        pred = F.softmax(pred, dim=1)
        target = torch.clamp(target.long(),1, self.num_classes) #确保标签在有效范围内
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()

        # 仅在已知区域计算
        # Unsqueeze masks to match pred shape (N, C, D, H, W)
        # known_mask = known_mask.float().clamp(0, 1).unsqueeze(1)
        # edge_mask = edge_mask.float().clamp(0, 1).unsqueeze(1)

        # 计算每个类别的Dice
        dice_loss = 0
        for c in range(pred.shape[1]):
            pred_c = pred[:, c]
            target_c = target_onehot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice)
        
        return dice_loss / pred.shape[1]


    def focal_loss(self, pred, target):
        """Focal损失"""
        # Reshape and filter out unknown regions
        # pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes) # N*D*H*W, C
        # target = target.view(-1) # N*D*H*W
        # known_mask = known_mask.view(-1) # N*D*H*W

        # # Select only elements from known regions
        # pred_known = pred[known_mask == 1]
        # target_known = target[known_mask == 1]

        # if pred_known.numel() == 0:
        #     return torch.tensor(0.0, device=pred.device)

        ce_loss = self.ce_loss(pred, target)
        # pt = torch.exp(-ce_loss)
        # focal_loss = 0.25 * (1 - pt) ** 2 * ce_loss

        return ce_loss #focal_loss.mean()


    def forward(self, seg_pred, seg_target, epoch):
        dice = self.dice_loss(seg_pred, seg_target)
        focal = self.focal_loss(seg_pred, seg_target)
        tversky = self.tversky(seg_pred, seg_target)
        # edge = self.boundary(seg_pred, seg_target, image)

         
        # dynamic
        if epoch < 30:
            # 早期：关注整体形状
            total = dice + 0.5 * focal + 0.3 * tversky
        else:
            # 后期：精化边界
            total = 0.7 * dice + 0.3 * focal + 0.2 * tversky
        
        # total = self.focal_loss(seg_pred, seg_target)

        return total, {
            'dice': dice.item(),
            'focal': focal.item(),
            'total': total.item()
        }

class RockCoreLoss(nn.Module):

    def __init__(
        self,
        num_classes:      int   = 4,
        lambda_dice:      float = 0.5,
        lambda_tversky:   float = 1.0,
        lambda_focal:     float = 0.5,
        lambda_interior:  float = 0.3,   
        tversky_alpha:    float = 0.7,
        tversky_beta:     float = 0.3,
        focal_gamma:      float = 2.0,
        ignore_margin:    int   = 2,     
        epsilon_inner:    float = 0.05,  
        epsilon_border:   float = 0.30, 
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.lambda_dice     = lambda_dice
        self.lambda_tversky  = lambda_tversky
        self.lambda_focal    = lambda_focal
        self.lambda_interior = lambda_interior
        self.alpha           = tversky_alpha
        self.beta            = tversky_beta
        self.gamma           = focal_gamma
        self.ignore_margin   = ignore_margin
        self.eps_inner       = epsilon_inner
        self.eps_border      = epsilon_border

        lap = torch.zeros(1, 1, 3, 3, 3)
        lap[0, 0, 1, 1, 1] = 6.0
        for idx in [(0,1,1),(2,1,1),(1,0,1),(1,2,1),(1,1,0),(1,1,2)]:
            lap[0, 0, idx[0], idx[1], idx[2]] = -1.0
        self.register_buffer('lap_kernel', lap)


    @torch.no_grad()
    def _interior_weight(self, target: torch.Tensor) -> torch.Tensor:
       
        target_f = target.float().unsqueeze(1)   # [B,1,D,H,W]

        lap_kernel = self.lap_kernel.to(target_f.device)
        edge = (F.conv3d(target_f, lap_kernel, padding=1).abs() > 0.5).float()

        for _ in range(self.ignore_margin):
            edge = (F.max_pool3d(edge, kernel_size=3, stride=1, padding=1) > 0).float()

        # 内部=1.0, 边界=epsilon_border
        weight = 1.0 - edge * (1.0 - self.eps_border)   # [B,1,D,H,W]
        return weight.squeeze(1)                          # [B,D,H,W]

    @torch.no_grad()
    def _soft_target(self, target: torch.Tensor,
                     interior_weight: torch.Tensor) -> torch.Tensor:
        """
        target:          [B, D, H, W] long
        interior_weight: [B, D, H, W] float in [eps_border, 1.0]
        return:          [B, C, D, H, W] float soft labels
        """
        B = target.shape[0]
        target_oh = F.one_hot(target, self.num_classes).float()  # [B,D,H,W,C]
        target_oh = target_oh.permute(0, 4, 1, 2, 3)            # [B,C,D,H,W]

     
        t = (interior_weight - self.eps_border) / (1.0 - self.eps_border + 1e-8)
        t = t.clamp(0.0, 1.0)                                    # [B,D,H,W]
        local_eps = self.eps_border + t * (self.eps_inner - self.eps_border)
        local_eps = local_eps.unsqueeze(1)                       # [B,1,D,H,W]

        # Soft label: confidence * one_hot + (1-confidence) / C 均匀分配
        soft = target_oh * (1.0 - local_eps) + local_eps / self.num_classes
        return soft   # [B,C,D,H,W]

    def dice_loss(self, pred: torch.Tensor, soft_target: torch.Tensor,
                  smooth: float = 1e-5) -> torch.Tensor:

        pred_soft = F.softmax(pred, dim=1)
        dims = (0, 2, 3, 4)
        inter = (pred_soft * soft_target).sum(dims)
        union = pred_soft.sum(dims) + soft_target.sum(dims)
        dice_per_class = (2 * inter + smooth) / (union + smooth)   # [C]

        # 小目标加权：按 GT 软体积倒数（体积越小权重越大，最大 10×）
        gt_volume   = soft_target.sum(dims).mean(0).clamp(min=1.0)  # [C]
        total_voxel = gt_volume.sum()
        weights     = (total_voxel / (gt_volume * self.num_classes)).clamp(max=10.0)
        return (weights * (1.0 - dice_per_class)).mean()

    def tversky_loss(self, pred: torch.Tensor, soft_target: torch.Tensor,
                     smooth: float = 1e-5) -> torch.Tensor:
 
        pred_soft = F.softmax(pred, dim=1)
        dims = (0, 2, 3, 4)
        TP = (pred_soft * soft_target).sum(dims)
        FP = (pred_soft * (1 - soft_target)).sum(dims)
        FN = ((1 - pred_soft) * soft_target).sum(dims)
        tversky = (TP + smooth) / (TP + self.alpha * FN + self.beta * FP + smooth)
        return (1.0 - tversky).mean()

    def focal_loss_interior(self, pred: torch.Tensor, target: torch.Tensor,
                            interior_weight: torch.Tensor) -> torch.Tensor:
   
        ce     = F.cross_entropy(pred, target, reduction='none')  # [B,D,H,W]
        pt     = torch.exp(-ce)
        focal  = (1 - pt) ** self.gamma * ce                      # [B,D,H,W]
        # 内部区域加权
        weighted = focal * interior_weight
        denom    = interior_weight.sum().clamp(min=1.0)
        return weighted.sum() / denom

    def interior_loss(self, pred: torch.Tensor, target: torch.Tensor,
                      interior_weight: torch.Tensor) -> torch.Tensor:
 
        ce       = F.cross_entropy(pred, target, reduction='none')  # [B,D,H,W]
        weighted = ce * interior_weight
        denom    = interior_weight.sum().clamp(min=1.0)
        return weighted.sum() / denom


    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred:   [B, C, D, H, W]  logits
        target: [B, D, H, W]     long class labels（含人工标注误差）
        """
      
        interior_weight = self._interior_weight(target)         # [B,D,H,W]
        soft_target     = self._soft_target(target, interior_weight)  # [B,C,D,H,W]

        L_dice     = self.dice_loss(pred, soft_target)
        L_tversky  = self.tversky_loss(pred, soft_target)
        L_focal    = self.focal_loss_interior(pred, target, interior_weight)
        L_interior = self.interior_loss(pred, target, interior_weight)

        total = (self.lambda_dice     * L_dice
               + self.lambda_tversky  * L_tversky
               + self.lambda_focal    * L_focal
               + self.lambda_interior * L_interior)

        return total,{
            'total':    total,
            'dice':     L_dice,
            'tversky':  L_tversky,
            'focal':    L_focal,
            'interior': L_interior,
        }
