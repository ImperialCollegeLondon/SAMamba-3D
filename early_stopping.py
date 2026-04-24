from copy import deepcopy

class EarlyStopping:
    """早停策略"""
    
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience (int): 在验证集性能没有提升的情况下等待多少个epoch后停止训练
            min_delta (float): 最小的性能提升阈值
            restore_best_weights (bool): 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_score, model, epoch):
        """
        检查是否应该早停
        
        Args:
            val_score (float): 验证集分数（越高越好）
            model (nn.Module): 模型
            epoch (int): 当前epoch
            
        Returns:
            bool: 是否应该早停
        """
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = deepcopy(model.state_dict())
                
        return self.early_stop
    
    def restore_best_weights_if_needed(self, model):
        """如果设置了恢复最佳权重，则恢复"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)