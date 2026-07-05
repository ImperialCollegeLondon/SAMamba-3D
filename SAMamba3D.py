import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List, Optional, Tuple, Dict


# =============================================================================
# utils
# =============================================================================

def _groupnorm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    actual_groups = min(num_groups, channels)
    while channels % actual_groups != 0 and actual_groups > 1:
        actual_groups -= 1
    return nn.GroupNorm(actual_groups, channels)


def _align_3d(feat: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
    if feat.shape[2:] == torch.Size(target_size):
        return feat
    return F.interpolate(feat, size=target_size, mode='trilinear', align_corners=True)

# =============================================================================
# Module 1  SAM_LoRA
# =============================================================================


class SAMBlockLoRABypass(nn.Module):
    def __init__(self, embed_dim: int, rank: int = 8, lora_alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.scale = lora_alpha / rank
        self.lora_A = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, embed_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, D, C] (SAM BHWDC) 或 [B, N, C]"""
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scale


class ECA3D(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)   # [B, C]
        w = self.conv(gap.unsqueeze(1)).squeeze(1)                    # [B, C]
        w = self.sigmoid(w).view(x.shape[0], x.shape[1], 1, 1, 1)
        return x * w


class BSEA_v2(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ch_attn = ECA3D(channels)
        self.sp_conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.norm = _groupnorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ch = self.ch_attn(x)
        sp_in = torch.cat([x_ch.mean(1, keepdim=True),
                           x_ch.max(1, keepdim=True)[0]], dim=1)
        sp_w = self.sp_conv(sp_in)
        return self.norm(x_ch * sp_w + x)


# =============================================================================
# Module 3 CrossScaleAdapter
# =============================================================================

class CrossScaleAdapter(nn.Module):

    def __init__(self, mamba_channels: int, sam_channels: int,
                 bottleneck_ratio: float = 0.25):
        super().__init__()
        mid = max(int(sam_channels * bottleneck_ratio), 32)

        self.local_path = nn.Sequential(
            nn.Conv3d(mamba_channels, mamba_channels, kernel_size=3,
                      padding=1, groups=mamba_channels, bias=False),
            nn.Conv3d(mamba_channels, mid, kernel_size=1, bias=False),
            _groupnorm(mid),
            nn.GELU(),
        )
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(mamba_channels, mid, bias=False),
            nn.GELU(),
        )
        self.path_gate = nn.Sequential(
            nn.Linear(mid * 2, 2, bias=True),
            nn.Softmax(dim=-1),
        )
        self.channel_proj = nn.Sequential(
            nn.Linear(mid, sam_channels, bias=False),
            nn.LayerNorm(sam_channels),
            nn.GELU(),
            nn.Linear(sam_channels, sam_channels, bias=False),
        )
 
        self.base_scale = nn.Parameter(torch.ones(sam_channels) * 1e-4)
        self.norm_out = nn.LayerNorm(sam_channels)

    def forward(self, mamba_feat: torch.Tensor, sam_feat: torch.Tensor,
                gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, H_s, W_s, D_s, C_sam = sam_feat.shape

        #  Mamba → SAM 
        mamba_hwdc = mamba_feat.permute(0, 1, 3, 4, 2).contiguous()  # [B,C,H,W,D]
        aligned = _align_3d(mamba_hwdc, (H_s, W_s, D_s))             # [B,C,H,W,D]

        # Local path
        local_feat = self.local_path(aligned)        # [B, mid, H, W, D]
        local_gap  = local_feat.mean(dim=[2, 3, 4])  # [B, mid]

    
        global_feat = self.global_path(mamba_feat)   # [B, mid]

        # Soft gate
        combined = torch.cat([local_gap, global_feat], dim=-1)   # [B, mid*2]
        weights  = self.path_gate(combined)                       # [B, 2]
        w_l, w_g = weights[:, 0:1], weights[:, 1:2]
        g_exp = global_feat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(local_feat)
        fused = (w_l.view(B,1,1,1,1) * local_feat
                 + w_g.view(B,1,1,1,1) * g_exp)                  # [B, mid, H, W, D]

        # Channel projection
        fused = fused.permute(0, 2, 3, 4, 1)                     # [B, H, W, D, mid]
        proj  = self.channel_proj(fused)                          # [B, H, W, D, C_sam]


        scale = self.base_scale
        if gamma is not None:
            # gamma: scalar 或 [B]
            if gamma.dim() == 1:
                scale = scale * gamma.view(B, 1)                
            else:
                scale = scale * gamma

        output = self.norm_out(sam_feat + proj * scale)
        return output


# =============================================================================
# Module 4: SAM→Mamba （SAMToMambaGate）
# =============================================================================

class SAMToMambaGate(nn.Module):
    """
    Input:
        sam_feat:   [B, H_s, W_s, D_s, C_sam]  (SAM BHWDC)
        mamba_feat: [B, C_m, D_m, H_m, W_m]
    Out:
        [B, C_m, D_m, H_m, W_m]  
    """

    def __init__(self, sam_channels: int, mamba_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(sam_channels, mamba_channels, kernel_size=1, bias=False),
            _groupnorm(mamba_channels),
            nn.GELU(),
        )
        # 初始化为 0：训练初期完全关闭，避免 SAM 未成熟特征干扰 Mamba
        self.layer_scale = nn.Parameter(torch.zeros(mamba_channels))
        self.norm = _groupnorm(mamba_channels)

    def forward(self, sam_feat: torch.Tensor,
                mamba_feat: torch.Tensor) -> torch.Tensor:
        B = mamba_feat.shape[0]
        target = mamba_feat.shape[2:]   # (D_m, H_m, W_m)

        # SAM BHWDC → BCDHW
        sam_bcdhw = sam_feat.permute(0, 4, 3, 1, 2).contiguous()
        sam_aligned = _align_3d(sam_bcdhw, target)
        sam_proj = self.proj(sam_aligned)                                  # [B, C_m, ...]

        scale = self.layer_scale.view(1, -1, 1, 1, 1)
        return self.norm(mamba_feat + sam_proj * scale)


# =============================================================================
# Module 5: BidirectionalBridge
# =============================================================================

class BidirectionalBridge(nn.Module):
    
    def __init__(self, mamba_channels: int, sam_channels: int):
        super().__init__()
        self.forward_adapter = CrossScaleAdapter(mamba_channels, sam_channels)
        self.reverse_gate    = SAMToMambaGate(sam_channels, mamba_channels)

    def forward(self, mamba_feat: torch.Tensor, sam_feat: torch.Tensor,
                gamma: Optional[torch.Tensor] = None,
                reverse_enabled: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mamba_feat:      [B, C_m, D_m, H_m, W_m]
            sam_feat:        [B, H_s, W_s, D_s, C_sam]
   
        Returns:
            sam_feat_new, mamba_feat_new
        """
        # Forward: Mamba → SAM
        sam_feat_new = self.forward_adapter(mamba_feat, sam_feat, gamma=gamma)

        # Reverse: SAM → Mamba
        if reverse_enabled:
            mamba_feat_new = self.reverse_gate(sam_feat_new, mamba_feat)
        else:
            mamba_feat_new = mamba_feat

        return sam_feat_new, mamba_feat_new


# =============================================================================
# Module 6: DACFM 
# =============================================================================

class DACFM(nn.Module):

    def __init__(self, sam_channels: int, mamba_channels: int, out_channels: int,
                 ctx_channels: int = 0):
        super().__init__()
        self.use_ctx = ctx_channels > 0

        self.mamba_proj = nn.Sequential(
            nn.Conv3d(mamba_channels, sam_channels, 1, bias=False),
            _groupnorm(sam_channels), nn.GELU(),
        )
        if self.use_ctx:
            self.ctx_proj = nn.Sequential(
                nn.Conv3d(ctx_channels, sam_channels, 1, bias=False),
                _groupnorm(sam_channels), nn.GELU(),
            )

        mid_s = max(sam_channels // 4, 16)
        self.sam_attn = nn.Sequential(
            nn.Linear(sam_channels, mid_s, bias=False), nn.GELU(),
            nn.Linear(mid_s, sam_channels, bias=False), nn.Sigmoid(),
        )
        self.mamba_attn = BSEA_v2(sam_channels)
        if self.use_ctx:
            self.ctx_attn = ECA3D(sam_channels)

        self.gate_fc = nn.Sequential(
            nn.Linear(sam_channels * 3, sam_channels, bias=True), nn.GELU(),
            nn.Linear(sam_channels, 3, bias=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv3d(sam_channels, out_channels, 1, bias=False),
            _groupnorm(out_channels), nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _groupnorm(out_channels),
        )
        self.shortcut = (nn.Conv3d(sam_channels, out_channels, 1, bias=False)
                         if sam_channels != out_channels else nn.Identity())
        self.act = nn.GELU()

    def forward(self, sam_feat: torch.Tensor, mamba_feat: torch.Tensor,
                ctx_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = sam_feat.shape[0]
        target = sam_feat.shape[2:]

        # Mamba 对齐 + 增强
        m = _align_3d(mamba_feat, target)
        m = self.mamba_proj(m)
        m_enh = self.mamba_attn(m)

        # SAM 通道注意力
        s_gap = sam_feat.mean(dim=[2, 3, 4])
        s_w   = self.sam_attn(s_gap).view(B, -1, 1, 1, 1)
        s_enh = sam_feat * s_w

    
        if self.use_ctx and ctx_feat is not None:
            c = _align_3d(ctx_feat, target)
            c = self.ctx_proj(c)
            c_enh = self.ctx_attn(c)
            c_gap = c_enh.mean(dim=[2, 3, 4])
        else:
            c_enh = torch.zeros_like(s_enh)
            c_gap = torch.zeros(B, s_gap.shape[1], device=sam_feat.device)

        # 三路 softmax 门控
        gate_input = torch.cat([s_gap, m_enh.mean(dim=[2, 3, 4]), c_gap], dim=-1)
        logits  = self.gate_fc(gate_input)
        weights = F.softmax(logits, dim=-1)
        α = weights[:, 0].view(B, 1, 1, 1, 1)
        β = weights[:, 1].view(B, 1, 1, 1, 1)
        γ = weights[:, 2].view(B, 1, 1, 1, 1)
        fused = α * s_enh + β * m_enh + γ * c_enh

        return self.act(self.out_proj(fused) + self.shortcut(fused))


# =============================================================================
# Module 7: Mamba （HierarchicalScaleOffset）
# =============================================================================

class HierarchicalScaleOffset(nn.Module):
    def __init__(self, channels: int, base_size: int = 4,
                 anisotropic_z_scale: float = 1.0):
        super().__init__()
     
        bias_init = torch.zeros(1, channels, base_size, base_size, base_size)
        if anisotropic_z_scale != 1.0:

            for c in range(channels):
                freq = (c + 1) / channels
                z_idx = torch.arange(base_size).float() * anisotropic_z_scale
                bias_init[0, c, :, base_size//2, base_size//2] = torch.sin(2 * math.pi * freq * z_idx / base_size)
        self.bias = nn.Parameter(bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D, H, W]"""
        bias = _align_3d(self.bias, x.shape[2:])
        return x + bias


class EarlyFusionStem(nn.Module):
   
    def __init__(self, in_chans: int, mamba_ch: int, stem_ch: int = 16):
        super().__init__()
        #  stem（3×3×3 conv）
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, stem_ch, kernel_size=3, padding=1, bias=False),
            _groupnorm(stem_ch), nn.GELU(),
            nn.Conv3d(stem_ch, stem_ch, kernel_size=3, padding=1, bias=False),
            _groupnorm(stem_ch), nn.GELU(),
        )
   
        self.mamba_proj = nn.Sequential(
            nn.Conv3d(mamba_ch, in_chans, kernel_size=1, bias=False),
            _groupnorm(in_chans), nn.GELU(),
        )
  
        self.fusion_scale = nn.Parameter(torch.ones(1) * 1e-3)
        self.stem_ch = stem_ch

    def forward(self, volume: torch.Tensor,
                mamba_s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        volume:    [B, in_chans, D, H, W]
        mamba_s0:  [B, mamba_ch, D/2, H/2, W/2]
        Returns:
            fused:     [B, in_chans, D, H, W]  →  PatchEmbed
            stem_feat: [B, stem_ch, D, H, W]   →  decoder HighResSkip
        """

        stem_feat = self.stem(volume)                          # [B, stem_ch, D, H, W]

        # Mamba s0 融合
        m = _align_3d(mamba_s0, volume.shape[2:])
        m = self.mamba_proj(m)
        fused = volume + self.fusion_scale.abs() * m        
        return fused, stem_feat

class HighResSkip(nn.Module):
    """
    为 decoder 最后一级上采样（D/2→D）提供原始分辨率 skip connection。

    动机（v5 新增）：
      decoder 的 up4 从 D/2 上采到 D，此时特征已经经过 4 次 down/upsample，
      高频边界信息严重丢失。直接将 EarlyFusionStem 的浅层特征引入，
      为最终输出提供"原始分辨率的孔隙边界先验"。

      这是 U-Net 架构中的经典思路，但在我们的框架中之前缺失了这一环。
      增加此模块后，decoder 的最后一级上采样可以感知原始体素级别的纹理。

    设计：
      stem_feat: [B, stem_ch, D, H, W]（来自 EarlyFusionStem）
      up4_feat:  [B, ch_up4, D, H, W]（来自 decoder up4 输出）
      → concat → 1×1 conv → [B, ch_out, D, H, W]

    skip_scale 初始化为 0，逐渐开放（类似 LayerScale 保护策略）。
    """

    def __init__(self, stem_ch: int, up4_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(stem_ch + up4_ch, out_ch, kernel_size=1, bias=False),
            _groupnorm(out_ch), nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _groupnorm(out_ch),
        )
        self.shortcut = nn.Conv3d(stem_ch + up4_ch, out_ch, 1, bias=False)
        self.act = nn.GELU()
        # LayerScale: init=0，保护 decoder 初期训练稳定性
        self.skip_scale = nn.Parameter(torch.zeros(out_ch))

    def forward(self, up4_feat: torch.Tensor, stem_feat: torch.Tensor) -> torch.Tensor:
        """
        up4_feat:  [B, up4_ch, D, H, W]
        stem_feat: [B, stem_ch, D, H, W]
        """
        # 对齐空间尺寸（防止 stride 不整除时 ±1 pixel 差异）
        if up4_feat.shape[2:] != stem_feat.shape[2:]:
            stem_feat = _align_3d(stem_feat, up4_feat.shape[2:])
        x = torch.cat([up4_feat, stem_feat], dim=1)
        skip_out = self.act(self.proj(x) + self.shortcut(x))
        # LayerScale 控制 skip 贡献强度
        scale = self.skip_scale.view(1, -1, 1, 1, 1)
        return up4_feat + skip_out * scale


# =============================================================================
# 模块 9: 3D Patch Embedding（stride=4，v5 关键修复）
# =============================================================================

class PatchEmbed3D(nn.Module):
    """
    Volume → SAM token。

    v5 关键修复：stride=8 → stride=4
    ─────────────────────────────────
    原因：岩心 CT 孔喉宽度通常 3-8 体素。
      stride=8 时：3体素孔隙 = 0.375 token → 完全被平均消失
      stride=4 时：3体素孔隙 = 0.75  token → 得到有效保留
                   5体素孔隙 = 1.25  token → 清晰表示

    代价：token 数量 8× 增加（D/8³ → D/4³）
    缓解：SAM 使用 window attention（非全局），配合 MambaGlobalController
          的 token routing 对非边界区域进行稀疏处理，实际计算量约增加 3-4×。

    输出格式：[B, D', H', W', C]，其中 D'=D/4, H'=H/4, W'=W/4
    """

    def __init__(self, in_chans: int = 1, embed_dim: int = 768,
                 kernel_size: Tuple = (4, 4, 4),
                 stride: Tuple = (4, 4, 4),
                 padding: Tuple = (0, 0, 0)):
        super().__init__()
        self.stride = stride
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D, H, W] → [B, D', H', W', embed_dim]"""
        x = self.proj(x)               # [B, embed_dim, D/4, H/4, W/4]
        x = x.permute(0, 2, 3, 4, 1)  # [B, D/4, H/4, W/4, embed_dim]
        return x


# =============================================================================
# 模块 10: Mamba Global Controller（核心新增模块）
# =============================================================================

class MambaGlobalController(nn.Module):

    def __init__(self, mamba_dims: List[int], sam_spatial_size: Tuple[int, int, int],
                 num_injection_layers: int = 4,
                 decoder_channels: List[int] = None):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        deep_ch = mamba_dims[-1]   # 384（s3 通道数）
        self.num_injection_layers = num_injection_layers
        self.decoder_channels = decoder_channels

        # 全局上下文向量
        self.context_pool = nn.AdaptiveAvgPool3d(1)   # → [B, deep_ch, 1,1,1]
        self.context_mlp  = nn.Sequential(
            nn.Linear(deep_ch, deep_ch, bias=False),
            nn.GELU(),
            nn.Linear(deep_ch, deep_ch, bias=False),
        )

        # (a) Token score map head: G → 空间路由分数
        # 用 s2 尺度（中分辨率，平衡语义和空间）生成 score map
        self.score_head = nn.Sequential(
            nn.Conv3d(mamba_dims[-2], 64, kernel_size=3, padding=1, bias=False),
            _groupnorm(64), nn.GELU(),
            nn.Conv3d(64, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # (b) 注入 gamma head: G → 每层注入强度
        self.gamma_head = nn.Sequential(
            nn.Linear(deep_ch, deep_ch // 2, bias=True),
            nn.GELU(),
            nn.Linear(deep_ch // 2, num_injection_layers, bias=True),
            nn.Sigmoid(),   # (0, 1)
        )

        # (c) FiLM 参数头: G → 所有 decoder stage 的 (γ, β)
        total_film = sum(c * 2 for c in decoder_channels)
        self.film_head = nn.Sequential(
            nn.Linear(deep_ch, deep_ch // 2, bias=True),
            nn.GELU(),
            nn.Linear(deep_ch // 2, total_film, bias=True),
        )

        self.sam_spatial = sam_spatial_size  # (D', H', W') of SAM tokens

    def forward(self, mamba_outs: List[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mamba_outs: [s0, s1, s2, s3],  [B, C_i, D_i, H_i, W_i]
        Returns:
            G:              [B, deep_ch]            
            score_map:      [B, 1, D', H', W']       
            gammas:         [B, num_injection_layers]  
            film_params:    [B, total_film_dim]       
        """
        s2, s3 = mamba_outs[-2], mamba_outs[-1]

        # 全局上下文
        G_raw = self.context_pool(s3).flatten(1)   # [B, deep_ch]
        G     = self.context_mlp(G_raw)            # [B, deep_ch]

        # (a) Score map from s2（
        score_map = self.score_head(s2)             # [B, 1, D_s2, H_s2, W_s2]
        if score_map.shape[2:] != torch.Size(self.sam_spatial):
            score_map = _align_3d(score_map, self.sam_spatial)

        # (b) dy gamma
        gammas = self.gamma_head(G)                 # [B, num_injection_layers]

        # (c) FiLM params
        film_params = self.film_head(G)             # [B, total_film_dim]

        return G, score_map, gammas, film_params


# =============================================================================
# 模块 11: FiLM （SAMDecodeModulator）
# =============================================================================

class SAMDecodeModulator(nn.Module):

    def __init__(self, decoder_channels: List[int]):
        super().__init__()
        self.decoder_channels = decoder_channels
        # 每个 stage 一个轻量 FiLM 投影头
        self.film_heads = nn.ModuleList([
            nn.Linear(1, ch * 2, bias=True)   # placeholder，实际参数来自 MambaGlobalController
            for ch in decoder_channels
        ])
        # 初始化：γ=1, β=0（恒等变换）
        for i, (head, ch) in enumerate(zip(self.film_heads, decoder_channels)):
            nn.init.zeros_(head.weight)
            # bias: 前 ch 个 → γ init=1, 后 ch 个 → β init=0
            head.bias.data[:ch]  = 1.0
            head.bias.data[ch:]  = 0.0

        # 用于将 film_params 解包为各 stage 参数的 offset 列表
        self._offsets = []
        offset = 0
        for ch in decoder_channels:
            self._offsets.append((offset, offset + ch * 2))
            offset += ch * 2

    def modulate(self, feat: torch.Tensor, film_params: torch.Tensor,
                 stage_idx: int) -> torch.Tensor:
        """
        feat:        [B, C, D, H, W]   — Mamba skip feature
        film_params: [B, total_film]   — 来自 MambaGlobalController
        stage_idx:   decoder stage 索引 (0=最深, 3=最浅)
        """
        B, C = feat.shape[:2]
        start, end = self._offsets[stage_idx]
        params = film_params[:, start:end]           # [B, 2C]
        gamma  = params[:, :C].view(B, C, 1, 1, 1)  # [B, C, 1, 1, 1]
        beta   = params[:, C:].view(B, C, 1, 1, 1)
        return gamma * feat + beta


# =============================================================================
# 模块 12: CoEncodingEncoder_v4
# =============================================================================

class CoEncodingEncoder_v4(nn.Module):

    def __init__(
        self,
        sam_img_encoder:     nn.Module,
        mamba_encoder:       nn.Module,
        patch_embed_3d:      nn.Module,
        mamba_dims:          List[int],             # [48, 96, 192, 384]
        sam_embed_dim:       int = 768,
        global_attn_indexes: List[int] = None,      # SAM global attn 层索引
        lora_rank:           int = 8,
        lora_alpha:          float = 16.0,
        out_channel:         int = 256,
        in_chans:            int = 1,
        anisotropic_z_scale: float = 1.0,           # 岩心 CT 各向异性参数
    ):
        super().__init__()

        self.mamba_encoder    = mamba_encoder
        self.sam_neck         = sam_img_encoder.neck
        self.patch_embed      = patch_embed_3d
        self.blocks           = sam_img_encoder.blocks
        self.num_blocks       = len(sam_img_encoder.blocks)
        num_scales            = len(mamba_dims)
        self.mamba_dims       = mamba_dims
        self.out_channel      = out_channel

        if global_attn_indexes is None:
            global_attn_indexes = [2, 5, 8, 11]

        self.shallow_injection = global_attn_indexes[:2]    # [2, 5]
        self.deep_bridge       = global_attn_indexes[2:]    # [8, 11]
        all_injection_points   = global_attn_indexes        # [2, 5, 8, 11]
        self.injection_to_scale = {
            layer: i for i, layer in enumerate(all_injection_points)
        }
        print(f"[CoEncoder v4] 浅层注入: {self.shallow_injection}, "
              f"中后层双向桥: {self.deep_bridge}")
        print(f"[CoEncoder v4] 注入点→Mamba尺度: {self.injection_to_scale}")

      
        self.early_fusion = EarlyFusionStem(in_chans=in_chans, mamba_ch=mamba_dims[0])

    
        self.scale_offsets = nn.ModuleList([
            HierarchicalScaleOffset(channels=mamba_dims[i], base_size=4,
                                    anisotropic_z_scale=anisotropic_z_scale)
            for i in range(num_scales)
        ])

        # ── Mamba Global Controller ──
        self.global_controller = MambaGlobalController(
            mamba_dims=mamba_dims,
            sam_spatial_size=(1, 1, 1), 
            num_injection_layers=len(all_injection_points),
            decoder_channels=mamba_dims[::-1],  
        )

 
        self.lora_bypasses = nn.ModuleList([
            SAMBlockLoRABypass(embed_dim=sam_embed_dim, rank=lora_rank, lora_alpha=lora_alpha)
            for _ in range(self.num_blocks)
        ])

      
        self.shallow_adapters = nn.ModuleDict()
        for layer_idx in self.shallow_injection:
            scale_idx = self.injection_to_scale[layer_idx]
            self.shallow_adapters[f"adapter_l{layer_idx}"] = CrossScaleAdapter(
                mamba_channels=mamba_dims[scale_idx],
                sam_channels=sam_embed_dim,
            )

  
        self.bidir_bridges = nn.ModuleDict()
        for layer_idx in self.deep_bridge:
            scale_idx = self.injection_to_scale[layer_idx]
            self.bidir_bridges[f"bridge_l{layer_idx}"] = BidirectionalBridge(
                mamba_channels=mamba_dims[scale_idx],
                sam_channels=sam_embed_dim,
            )

     
        self.dacfm_nodes = nn.ModuleDict()
        for i, layer_idx in enumerate(self.deep_bridge):
            scale_idx = self.injection_to_scale[layer_idx]
            self.dacfm_nodes[f"dacfm_l{layer_idx}"] = DACFM(
                sam_channels=sam_embed_dim,
                mamba_channels=mamba_dims[scale_idx],
                out_channels=out_channel,
                ctx_channels=out_channel if i > 0 else 0,
            )

    
        self.shallow_dacfm = nn.ModuleDict()
        for layer_idx in self.shallow_injection:
            scale_idx = self.injection_to_scale[layer_idx]
            self.shallow_dacfm[f"dacfm_l{layer_idx}"] = DACFM(
                sam_channels=sam_embed_dim,
                mamba_channels=mamba_dims[scale_idx],
                out_channels=out_channel,
                ctx_channels=0,
            )

        self.reverse_enabled = False

    def set_training_stage(self, stage: str):
      
        self.reverse_enabled = (stage in ('B', 'C'))
        print(f"[CoEncoder v5] Training stage={stage}, "
              f"reverse_enabled={self.reverse_enabled}")

    def forward(self, volume_3d: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor],
                           torch.Tensor, torch.Tensor]:

        mamba_outs_raw = self.mamba_encoder(volume_3d)
      
        mamba_outs = [self.scale_offsets[i](mamba_outs_raw[i])
                      for i in range(len(mamba_outs_raw))]

        volume_fused, stem_feat = self.early_fusion(volume_3d, mamba_outs[0])
        # stem_feat: [B, stem_ch, D, H, W]  原始分辨率浅层特征
        # volume_fused: [B, C, D, H, W] → x_sam: [B, D/4, H/4, W/4, C_sam]
        x_sam = self.patch_embed(volume_fused)
        sam_spatial = (x_sam.shape[1], x_sam.shape[2], x_sam.shape[3])  # (D/4, H/4, W/4)
        self.global_controller.sam_spatial = sam_spatial

        G, score_map, gammas, film_params = self.global_controller(mamba_outs)
        intermediate_features = []   
        ctx_feat = None

        for i, blk in enumerate(self.blocks):
            # SAM Block 
            x_sam = blk(x_sam)
            # LoRA 
            x_sam = x_sam + self.lora_bypasses[i](x_sam)

       
            if i in self.shallow_injection:
                scale_idx  = self.injection_to_scale[i]
                mamba_feat = mamba_outs[scale_idx]       # [B, C_m, D_m, H_m, W_m]
                gamma_i    = gammas[:, scale_idx]        # [B]

                # Forward adapter
                x_sam = self.shallow_adapters[f"adapter_l{i}"](
                    mamba_feat, x_sam, gamma=gamma_i)

                sam_bcdhw = x_sam.permute(0, 4, 3, 1, 2).contiguous()  # [B,C,D/8,H/8,W/8]
                mamba_target = mamba_feat.shape[2:]                      # (D_m, H_m, W_m)
                sam_for_dacfm = _align_3d(sam_bcdhw, mamba_target)    
                fused = self.shallow_dacfm[f"dacfm_l{i}"](
                    sam_for_dacfm, mamba_feat)                           # [B,C_out,D_m,H_m,W_m]
                intermediate_features.append(fused)

        
            elif i in self.deep_bridge:
                scale_idx  = self.injection_to_scale[i]
                mamba_feat = mamba_outs[scale_idx]       # [B, C_m, D_m, H_m, W_m]
                gamma_i    = gammas[:, scale_idx]        # [B]

            
                x_sam, mamba_feat_updated = self.bidir_bridges[f"bridge_l{i}"](
                    mamba_feat, x_sam,
                    gamma=gamma_i,
                    reverse_enabled=self.reverse_enabled,
                )
             
                mamba_outs[scale_idx] = mamba_feat_updated
                sam_bcdhw = x_sam.permute(0, 4, 3, 1, 2).contiguous()  # [B,C,D/4,H/4,W/4]
                mamba_target = mamba_feat_updated.shape[2:]              # (D_m, H_m, W_m)
                sam_for_dacfm = _align_3d(sam_bcdhw, mamba_target)   
                fused = self.dacfm_nodes[f"dacfm_l{i}"](
                    sam_for_dacfm, mamba_feat_updated,
                    ctx_feat=ctx_feat)                                   #  [B,C_out,D_m,H_m,W_m]
                intermediate_features.append(fused)
                ctx_feat = fused  

        x_bcdhw = x_sam.permute(0, 4, 3, 1, 2).contiguous()
        x_out = self.sam_neck(x_bcdhw)    # [B, C_out, D/4, H/4, W/4]  (stride=4)

        return x_out, intermediate_features, mamba_outs, film_params, stem_feat


class _UpBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _groupnorm(out_channels), nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _groupnorm(out_channels),
        )
        self.residual = nn.Conv3d(out_channels, out_channels, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor,
                target_size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        x = self.up(x)
        if target_size is not None and x.shape[2:] != torch.Size(target_size):
            x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=True)
        return self.act(self.conv(x) + self.residual(x))


class _FuseBlock(nn.Module):
    """
    Skip connection 
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            _groupnorm(out_channels), nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _groupnorm(out_channels), nn.GELU(),
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


# =============================================================================
# module 14: HybridCoDecoder
# =============================================================================

class HybridCoDecoder_v5(nn.Module):
   
    def __init__(self, sam_channels: int = 256,
                 mamba_dims: List[int] = None,
                 num_classes: int = 4,
                 out_chans: int = 256,
                 decoder_channels: List[int] = None,
                 stem_ch: int = 16):
        super().__init__()
        if mamba_dims is None:
            mamba_dims = [32,64,128,256]
        if decoder_channels is None:
            decoder_channels = mamba_dims[::-1]  

        self.film_modulator = SAMDecodeModulator(decoder_channels)
        hidden = decoder_channels[0]   # 256

        # ── 融合起点：SAM neck(D/4) 下采到 D/16 + FiLM(s3) ──
        self.sam_to_bottleneck = nn.Sequential(
            nn.Conv3d(sam_channels, hidden, 1, bias=False),
            _groupnorm(hidden), nn.GELU(),
        )
        self.scale3_proj  = nn.Sequential(
            nn.Conv3d(mamba_dims[3], hidden, 1, bias=False),
            _groupnorm(hidden), nn.GELU(),
        )
        self.inter_proj3 = nn.Conv3d(out_chans, hidden, 1, bias=False)

        # ── D/16 → D/8：up1 + FiLM(s2) + inter2 → fuse2 ──
        self.up1         = _UpBlock(hidden * 2, hidden)
        self.scale2_proj = nn.Conv3d(mamba_dims[2], hidden, 1, bias=False)
        self.inter_proj2 = nn.Conv3d(out_chans, hidden, 1, bias=False)
        self.fuse2       = _FuseBlock(hidden * 2, hidden)

        # ── D/8 → D/4：up2 + FiLM(s1) + inter1 → fuse1 ──
        # stride=4 后 SAM neck = D/4 = scale1 分辨率，inter1 天然对齐
        self.up2         = _UpBlock(hidden, hidden // 2)
        self.scale1_proj = nn.Conv3d(mamba_dims[1], hidden // 2, 1, bias=False)
        self.inter_proj1 = nn.Conv3d(out_chans, hidden // 2, 1, bias=False)
        self.fuse1       = _FuseBlock(hidden, hidden // 2)

        # ── D/4 → D/2：up3 + FiLM(s0) + inter0 → fuse0 ──
        self.up3         = _UpBlock(hidden // 2, hidden // 4)
        self.scale0_proj = nn.Conv3d(mamba_dims[0], hidden // 4, 1, bias=False)
        self.inter_proj0 = nn.Conv3d(out_chans, hidden // 4, 1, bias=False)
        self.fuse0       = _FuseBlock(hidden // 2, hidden // 4)

        # ── D/2 → D：up4 ──
        self.up4 = _UpBlock(hidden // 4, 32)

     
        self.hi_skip = HighResSkip(
            stem_ch=stem_ch,
            up4_ch=32,
            out_ch=32,
        )

        self.final_conv = nn.Sequential(
            nn.Conv3d(32, 32, 3, padding=1, bias=False),
            _groupnorm(32), nn.GELU(),
            nn.Conv3d(32, num_classes, 1),
        )

    def forward(self, sam_feat: torch.Tensor,
                intermediate_features: List[torch.Tensor],
                mamba_outs: List[torch.Tensor],
                film_params: torch.Tensor,
                stem_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sam_feat:              [B, C_out, D/4, H/4, W/4]  SAM neck (stride=4)
            intermediate_features: [inter0@D/2, inter1@D/4, inter2@D/8, inter3@D/16]
            mamba_outs:            [s0@D/2, s1@D/4, s2@D/8, s3@D/16]
            film_params:           [B, total_film_dim]
            stem_feat:             [B, stem_ch, D, H, W]  原始分辨率浅层特征
        Returns:
            [B, num_classes, D, H, W]
        """
        scale0, scale1, scale2, scale3 = mamba_outs
        inter0, inter1, inter2, inter3 = intermediate_features

       
        target_d16 = scale3.shape[2:]
        sam_d16 = _align_3d(sam_feat, target_d16)              # D/4 → D/16
        sam_d16 = self.sam_to_bottleneck(sam_d16)              # [B, 256, D/16...]

        s3_mod = self.film_modulator.modulate(scale3, film_params, stage_idx=0)
        s3     = self.scale3_proj(s3_mod)
        inter3_aligned = _align_3d(inter3, target_d16)         
        s3     = s3 + self.inter_proj3(inter3_aligned)
        x      = torch.cat([sam_d16, s3], dim=1)               # [B, 512, D/16...]

        #  D/16 → D/8 
        target_d8 = scale2.shape[2:]
        x      = self.up1(x, target_size=target_d8)            # [B, 256, D/8...]
        s2_mod = self.film_modulator.modulate(scale2, film_params, stage_idx=1)
        s2     = self.scale2_proj(s2_mod)
        inter2_aligned = _align_3d(inter2, target_d8)
        s2     = s2 + self.inter_proj2(inter2_aligned)
        x      = self.fuse2(torch.cat([x, s2], dim=1))         # [B, 256, D/8...]

        #  D/8 → D/4 
        target_d4 = scale1.shape[2:]
        x      = self.up2(x, target_size=target_d4)            # [B, 128, D/4...]
        s1_mod = self.film_modulator.modulate(scale1, film_params, stage_idx=2)
        s1     = self.scale1_proj(s1_mod)
        inter1_aligned = _align_3d(inter1, target_d4)
        s1     = s1 + self.inter_proj1(inter1_aligned)
        x      = self.fuse1(torch.cat([x, s1], dim=1))         # [B, 128, D/4...]

        #  D/4 → D/2 
        target_d2 = scale0.shape[2:]
        x      = self.up3(x, target_size=target_d2)            # [B, 64, D/2...]
        s0_mod = self.film_modulator.modulate(scale0, film_params, stage_idx=3)
        s0     = self.scale0_proj(s0_mod)
        inter0_aligned = _align_3d(inter0, target_d2)
        s0     = s0 + self.inter_proj0(inter0_aligned)
        x      = self.fuse0(torch.cat([x, s0], dim=1))         # [B, 64, D/2...]

        #  D/2 → D 
        original_size = tuple(s * 2 for s in scale0.shape[2:])
        x = self.up4(x)                                        # [B, 32, D...]
        if x.shape[2:] != torch.Size(original_size):
            x = F.interpolate(x, size=original_size, mode='trilinear', align_corners=True)

        #  HighResSkip 
        x = self.hi_skip(x, stem_feat)                         # [B, 32, D, H, W]

        return self.final_conv(x)                              # [B, num_classes, D, H, W]


# =============================================================================
# SAM_Mamba_3D_CoEncoding
# =============================================================================

class SAM_Mamba_3D_CoEncoding(nn.Module):

    def __init__(
        self,
        model_type:          str   = 'vit_b',
        mamba_config:        dict  = None,
        num_classes:         int   = 4,
        in_chans:            int   = 1,
        out_chans:           int   = 256,
        lora_rank:           int   = 8,
        lora_alpha:          float = 16.0,
        anisotropic_z_scale: float = 1.0,
        stem_ch:             int   = 16,   
    ):
        super().__init__()
        self.num_classes = num_classes
        self.stem_ch     = stem_ch

        # ── Mamba Encoder ──
        from mamba_encoder import MambaEncoder
        if mamba_config is None:
            mamba_config = {
                'in_chans': in_chans,
                'depths': [2, 2, 2, 2],
                'dims': [48, 96, 192, 384],
                'drop_path_rate': 0.1,
                'out_indices': [0, 1, 2, 3],
            }
        mamba_dims    = mamba_config['dims']
        mamba_encoder = MambaEncoder(**mamba_config)

        # ── SAM Image Encoder ──
        from AdapterSAM.image_encoder_3d import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        if model_type == 'vit_l':
            sam_depth, sam_embed_dim = 24, 1024
            global_attn_indexes = [5, 11, 17, 23]
        else:
            sam_depth, sam_embed_dim = 12, 768
            global_attn_indexes = [2, 5, 8, 11]

        sam_img_encoder = ImageEncoderViT_3d(
            depth=sam_depth,
            embed_dim=sam_embed_dim,
            in_chans=48,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=16 if model_type == 'vit_l' else 12,
            patch_size=16,
            qkv_bias=True,
            use_abs_pos=False,
            use_rel_pos=False,
            rel_pos_zero_init=False,
            global_attn_indexes=global_attn_indexes,
            window_size=16,
            cubic_window_size=8,
            out_chans=out_chans,
        )

        # ── 3D Patch Embedding（stride=4）──
        patch_embed_3d = PatchEmbed3D(
            in_chans=in_chans,
            embed_dim=sam_embed_dim,
            kernel_size=(4, 4, 4),
            stride=(4, 4, 4),
        )

        self.co_encoder = CoEncodingEncoder_v4(
            sam_img_encoder     = sam_img_encoder,
            mamba_encoder       = mamba_encoder,
            patch_embed_3d      = patch_embed_3d,
            mamba_dims          = mamba_dims,
            sam_embed_dim       = sam_embed_dim,
            global_attn_indexes = global_attn_indexes,
            lora_rank           = lora_rank,
            lora_alpha          = lora_alpha,
            out_channel         = out_chans,
            in_chans            = in_chans,
            anisotropic_z_scale = anisotropic_z_scale,
        )

        self.decoder = HybridCoDecoder_v5(
            sam_channels     = out_chans,
            mamba_dims       = mamba_dims,
            num_classes      = num_classes,
            out_chans        = out_chans,
            decoder_channels = mamba_dims[::-1],
            stem_ch          = stem_ch,
        )

        self._current_stage = 'A'

    def set_training_stage(self, stage: str, sam_checkpoint: Optional[str] = None):
        
        assert stage in ('A', 'B', 'C'), f"stage must be A/B/C，got {stage}"
        self._current_stage = stage
        self.co_encoder.set_training_stage(stage)

        # force freeze all parameters first
        for p in self.parameters():
            p.requires_grad = False

        always_on = [
            self.co_encoder.mamba_encoder,
            self.co_encoder.early_fusion,
            self.co_encoder.patch_embed,
            self.co_encoder.scale_offsets,
            self.co_encoder.global_controller,
            self.co_encoder.shallow_adapters,
            self.co_encoder.bidir_bridges,
            self.co_encoder.dacfm_nodes,
            self.co_encoder.shallow_dacfm,
            self.co_encoder.sam_neck,
            self.decoder,
        ]
        for m in always_on:
            for p in m.parameters():
                p.requires_grad = True
        
        # Stage A: SAM Block Attn/MLP 全冻结，LayerNorm 冻结，LoRA 冻结，reverse gate 关闭
        if stage == 'A':
            for blk in self.co_encoder.blocks:
                for m in blk.modules():
                    if isinstance(m, nn.LayerNorm):
                        for p in m.parameters():
                            p.requires_grad = False
            
            for p in self.co_encoder.lora_bypasses.parameters():
                p.requires_grad = False
            self.reverse_enabled = False

        # ── Stage B: SAM LayerNorm + adapters + reverse gate 
        if stage in ('B', 'C'):
            for blk in self.co_encoder.blocks:
                for m in blk.modules():
                    if isinstance(m, nn.LayerNorm):
                        for p in m.parameters():
                            p.requires_grad = True
     
                if hasattr(blk, 'adapter') and blk.adapter is not None:
                    for p in blk.adapter.parameters():
                        p.requires_grad = True
             # ── Stage C: SAM LayerNorm + LoRA + reverse gate ──
            if stage =='C':
                for p in self.co_encoder.lora_bypasses.parameters():
                    p.requires_grad = True


        if stage == 'A' and sam_checkpoint is not None:
            self._load_sam_weights(sam_checkpoint)

        # self._print_param_stats()

    def _load_sam_weights(self, checkpoint: str):
        import os
        if not os.path.exists(checkpoint):
            print(f"[Warning] SAM checkpoint not found: {checkpoint}")
            return
        from segment_anything import sam_model_registry
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        state_dict = sam.image_encoder.state_dict()
        missing, unexpected = self.co_encoder.load_state_dict(
            {f"blocks.{k}": v for k, v in state_dict.items()},
            strict=False,
        )
        del sam
        torch.cuda.empty_cache()
        print(f"[v5] SAM weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # def _print_param_stats(self):
    #     total     = sum(p.numel() for p in self.parameters())
    #     trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     lora_p    = sum(p.numel() for p in self.co_encoder.lora_bypasses.parameters())
    #     ctrl_p    = sum(p.numel() for p in self.co_encoder.global_controller.parameters())
    #     bidir_p   = sum(p.numel() for p in self.co_encoder.bidir_bridges.parameters())
    #     hi_p      = sum(p.numel() for p in self.decoder.hi_skip.parameters())
    #     print(
    #         f"\n[v5 Params | Stage={self._current_stage}]\n"
    #         f"  Total:            {total/1e6:.2f}M\n"
    #         f"  Trainable:        {trainable/1e6:.2f}M ({100*trainable/total:.1f}%)\n"
    #         f"  LoRA bypasses:    {lora_p/1e6:.3f}M\n"
    #         f"  GlobalController: {ctrl_p/1e6:.3f}M\n"
    #         f"  BidirBridges:     {bidir_p/1e6:.3f}M\n"
    #         f"  HighResSkip:      {hi_p/1e6:.3f}M\n"
    #     )

    def forward_stage1(self, volume_3d: torch.Tensor,
                       gt_masks=None, training: bool = False) -> dict:
    
        sam_feat, intermediate_features, mamba_outs, film_params, stem_feat = \
            self.co_encoder(volume_3d)

        output = self.decoder(
            sam_feat, intermediate_features, mamba_outs, film_params, stem_feat)

        if training:
            return {
                'final_output':         output,
                'intermediate_outputs': intermediate_features,
                'film_params':          film_params,
            }
        return  output




if __name__ == "__main__":
    
    mamba_config = {
        'in_chans': 1,
        'depths': [2, 2, 2, 2],
        'dims':  [48, 96, 192, 384],
        'drop_path_rate': 0.1,
        'out_indices': [0, 1, 2, 3],
    }


    model = SAM_Mamba_3D_CoEncoding(
        model_type          = 'vit_b',
        mamba_config        = mamba_config,
        num_classes         = 4,       
        out_chans           = 256,
        lora_rank           = 8,
        lora_alpha          = 16.0,
        anisotropic_z_scale = 1.0,   
        stem_ch             = 16,
    ).cuda()

    input_volume = torch.randn(1, 1, 96,96,96).cuda()  # [B, C, D, H, W]
    output = model.forward_stage1(input_volume, training=True)
    print("Output shape:", output['final_output'].shape) # [B, num_classes, D, H, W]

    model.set_training_stage('A', sam_checkpoint='sam_vit_b_01ec64.pth')
    # ... 20-30 epochs: Mamba + Adapters + Decoder

    model.set_training_stage('B')
    # ... 50-60 epochs: add SAM LoRA + LayerNorm + Reverse gate

    

