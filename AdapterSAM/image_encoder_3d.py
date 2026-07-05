import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from segment_anything.modeling.common import LayerNorm2d, MLPBlock
from segment_anything.modeling.image_encoder import Attention, PatchEmbed, window_unpartition
import math
class Adapter(nn.Module):
    def __init__(
            self,
            input_dim,
            mid_dim,
            
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.conv = nn.Conv3d(in_channels = mid_dim, out_channels = mid_dim, kernel_size=3, padding=1, groups=mid_dim)
        self.linear2 = nn.Linear(mid_dim, input_dim)

    def forward(self, features):
        out = self.linear1(features)
        out = F.relu(out)
        out = out.permute(0, 4, 1, 2, 3)
        out = self.conv(out)
        out = out.permute(0, 2, 3, 4, 1)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = features + out
        return out

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class ImageEncoderViT_3d(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        patch_depth: int=32,
        in_chans: int = 48,
        embed_dim: int = 1024,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        cubic_window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        num_slice = 1 
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.depth = depth

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
            self.depth_embed = nn.Parameter(
                torch.ones(1, patch_depth, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
    
            block = Block_3d(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=cubic_window_size,
                    res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                    shift=cubic_window_size // 2 if i % 2 == 0 else 0
                )
    
            
            self.blocks.append(block)

        self.neck_3d = nn.ModuleList()
        for i in range(4):
            self.neck_3d.append(nn.Sequential(
                nn.Conv3d(embed_dim, out_chans, 1, bias=False),
                LayerNorm3d(out_chans),
                nn.Conv3d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm3d(out_chans),
            )) #旧版原始neck


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # if self.pos_embed is not None:
        #     pos_embed = F.avg_pool2d(self.pos_embed.permute(0,3,1,2), kernel_size=2).permute(0,2,3,1).unsqueeze(3)
        #     pos_embed = pos_embed + (self.depth_embed.unsqueeze(1).unsqueeze(1))
        #     x = x + pos_embed

        idx = 0
        feature_list = []
        
        for blk in self.blocks[:self.depth//2]:
            x = blk(x)
            idx += 1
            if idx % (self.depth//4) == 0 and idx != self.depth:
                # print(f"SAM block3d shape: {x.shape}")
                feature_list.append(self.neck_3d[idx//(self.depth//4)-1](x.permute(0, 4, 1, 2, 3)))

                
        for blk in self.blocks[self.depth//2:self.depth]:
            x = blk(x)
            idx += 1
            if idx % (self.depth//4) == 0 and idx != self.depth:
                feature_list.append(self.neck_3d[idx//(self.depth//4)-1](x.permute(0, 4, 1, 2, 3)))

        x = self.neck_3d[-1](x.permute(0, 4, 1, 2, 3))
        feature_list.append(x)

        return x, feature_list

class ImageEncoderViT_3d_v2(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            patch_depth: int=32,
            in_chans: int = 48,
            embed_dim: int = 1024,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            cubic_window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            num_slice = 1 
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.depth = depth

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
            self.depth_embed = nn.Parameter(
                torch.ones(1, patch_depth, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
    
            block = Block_3d(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=cubic_window_size,
                    res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                    shift=cubic_window_size // 2 if i % 2 == 0 else 0
                )
    
            
            self.blocks.append(block)

        # self.neck_3d = nn.ModuleList()
        # for i in range(4):
        #     self.neck_3d.append(nn.Sequential(
        #         nn.Conv3d(embed_dim, out_chans, 1, bias=False),
        #         LayerNorm3d(out_chans),
        #         nn.Conv3d(
        #             out_chans,
        #             out_chans,
        #             kernel_size=3,
        #             padding=1,
        #             bias=False,
        #         ),
        #         LayerNorm3d(out_chans),
        #     )) 旧版原始neck

        self.neck = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            # nn.LayerNorm(out_chans),
            LayerNorm3d(out_chans),
            nn.Conv3d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
            # nn.LayerNorm(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # if self.pos_embed is not None:
        #     pos_embed = F.avg_pool2d(self.pos_embed.permute(0,3,1,2), kernel_size=2).permute(0,2,3,1).unsqueeze(3)
        #     pos_embed = pos_embed + (self.depth_embed.unsqueeze(1).unsqueeze(1))
        #     x = x + pos_embed

        idx = 0
        feature_list = []
        
        for blk in self.blocks[:self.depth//2]:
            x = blk(x)
            idx += 1
            if idx % (self.depth//4) == 0 and idx != self.depth:
                # print(f"SAM block3d shape: {x.shape}")
                feature_list.append(self.neck_3d[idx//(self.depth//4)-1](x.permute(0, 4, 1, 2, 3)))

                
        for blk in self.blocks[self.depth//2:self.depth]:
            x = blk(x)
            idx += 1
            if idx % (self.depth//4) == 0 and idx != self.depth:
                feature_list.append(self.neck_3d[idx//(self.depth//4)-1](x.permute(0, 4, 1, 2, 3)))

        x = self.neck_3d[-1](x.permute(0, 4, 1, 2, 3))
        feature_list.append(x)

        return x, feature_list



class Block_3d(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            res_size = None,
            shift = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=(window_size, window_size, window_size),
            res_size=(res_size, res_size, res_size),
        )
        self.shift_size = shift
        # if self.shift_size > 0:
            # H, W, D = 36,36,36 #输入维度/2
            # img_mask = torch.zeros((1, H, W, D, 1))
            # h_slices = (slice(0, -window_size),
            #             slice(-window_size, -self.shift_size),
            #             slice(-self.shift_size, None))
            # w_slices = (slice(0, -window_size),
            #             slice(-window_size, -self.shift_size),
            #             slice(-self.shift_size, None))
            # d_slices = (slice(0, -window_size),
            #             slice(-window_size, -self.shift_size),
            #             slice(-self.shift_size, None))
            # cnt = 0
            # for h in h_slices:
            #     for w in w_slices:
            #         for d in d_slices:
            #             img_mask[:, h, w, d, :] = cnt
            #             cnt += 1
            # mask_windows = window_partition(img_mask, window_size)[0]
            # mask_windows = mask_windows.view(-1, window_size * window_size * window_size)
            # attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0,
            #                                                                              float(0.0))
            # print(f"Mask shape {attn_mask.shape}") (-1,8^3,8^3)

        # else:
        #     attn_mask = None
        # self.register_buffer("attn_mask", attn_mask)
        self._mask_cache: dict = {}
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size
        self.adapter = Adapter(input_dim=dim, mid_dim = dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        shortcut = x
        x = self.norm1(x)

        # print(f"输入维度：{x.shape}") 1,h/2,w/2,d/2,c
        # ── 动态计算当前输入尺寸下的 attn_mask，而不是用初始化时的静态 buffer ──
        B, H, W, D, C = x.shape
        # ── 动态获取/计算 attn_mask ──────────────────────────────────────────
        # 以 (H, W, D) 为 key 缓存 mask，支持训练/推理使用不同 patch 尺寸。
        # shift_size == 0 时（非 shifted block）不需要 mask，直接用 None。
        if self.shift_size > 0:
            cache_key = (H, W, D)
            if cache_key not in self._mask_cache:
                self._mask_cache[cache_key] = self._compute_attn_mask(
                    input_size=(H, W, D),
                    window_size=self.window_size,
                    shift_size=self.shift_size,
                    device=x.device,
                )
            attn_mask = self._mask_cache[cache_key]
            # 确保 mask 与输入在同一设备（多 GPU 或 CPU↔GPU 切换时）
            if attn_mask.device != x.device:
                attn_mask = attn_mask.to(x.device)
                self._mask_cache[cache_key] = attn_mask
        else:
            attn_mask = None
       
        # Window partition
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1,2,3))
        x, pad_hw = window_partition(x, self.window_size)
        # x: [nW*B, window_size, window_size, window_size, C]
        # reshape 为 attention 期望的 [nW*B, ws³, C]
        ws3 = self.window_size ** 3
        x = x.view(-1, ws3, C)
        x = self.attn(x, mask=attn_mask)
        
        # Reverse window partition
        x = x.view(-1, self.window_size, self.window_size, self.window_size, C)

        # Reverse window partition（去除 padding，恢复原始空间尺寸）
        x = window_unpartition(x, self.window_size, pad_hw, (H, W, D))

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size, self.shift_size),
                dims=(1, 2, 3)
            )

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
    
    def _compute_attn_mask(self,input_size, window_size, shift_size, device):
            """动态计算任意输入尺寸下的 shifted window attention mask"""
            H, W, D = input_size
            ws = window_size
            
            # ── 对输入空间按 shift 边界划分区域并编号 ──
            # 每个维度有 3 个切片区域：[0:-ws], [-ws:-shift], [-shift:None]
            # 3D 下共 3³=27 个区域，用 0~26 的整数标记
            img_mask = torch.zeros((1, H, W, D, 1), device=device)

            h_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
            d_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    for d in d_slices:
                        img_mask[:, h, w, d, :] = cnt
                        cnt += 1
            # img_mask: [1, H, W, D, 1]，每个位置存储其所属区域编号

            # ── window partition：将 mask 按 window 切分 ──
            # window_partition 内部处理了 padding，确保 H/W/D 不是 ws 整数倍时也正确
            mask_windows, _ = window_partition(img_mask, ws)
            # mask_windows: [nW, ws, ws, ws, 1]
            mask_windows = mask_windows.view(-1, ws * ws * ws)
            # mask_windows: [nW, ws³]，每个 token 的区域编号

            # ── 计算 attention mask ──
            # 同一 window 内，若两个 token 区域编号不同，则禁止互相 attend
            # unsqueeze 后做广播减法：[nW, 1, ws³] - [nW, ws³, 1] → [nW, ws³, ws³]
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # 差值非零 → 不同区域 → 填 -100（softmax 后趋近 0，即禁止）
            # 差值为零 → 同一区域 → 填 0（正常 attend）
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

            return attn_mask  # [nW, ws³, ws³]


class Attention_3d(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
            res_size = None
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * res_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * res_size[1] - 1, head_dim))
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * res_size[2] - 1, head_dim))
            self.lr = nn.Parameter(torch.tensor(1.))

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, H, W, D, _ = x.shape
        # print(f"B, H, W, D, C {x.shape}")
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W * D, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #q, k, v = qkv.reshape(3, B * self.num_heads, H * W * D, -1).unbind(0)
        q_sub = q.reshape(B * self.num_heads, H * W * D, -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q_sub, self.rel_pos_h, self.rel_pos_w, self.rel_pos_d, (H, W, D), (H, W, D), self.lr)
            attn = attn.reshape(B, self.num_heads, H * W * D, -1)
        if mask is None:
            attn = attn.softmax(dim=-1)
        else:
            nW = mask.shape[0]
            # print("nW ",nW)
            attn = attn.view(B // nW, nW, self.num_heads, H*W*D, H*W*D) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, H*W*D, H*W*D)
            attn = attn.softmax(dim=-1)
        
        # print(f"attn, v shape: {attn.shape}, {v.shape}")
        x = (attn @ v).view(B, self.num_heads, H, W, D, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, H, W, D, -1)
        x = self.proj(x)

        return x

# ── image_encoder_3d.py ──

class WindowAttention3D(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
            res_size=None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size: Window spatial size (H, W, D) for relative pos embedding shape.
            res_size: Resolution size for relative pos embedding initialization.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # ── 原版缺少这两个成员，forward 中引用时会报 AttributeError ──
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Identity()   # 推理阶段 dropout=0，用 Identity 占位；
                                          # 如需训练期 dropout 可改为 nn.Dropout(p)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * res_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * res_size[1] - 1, head_dim))
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * res_size[2] - 1, head_dim))
            self.lr = nn.Parameter(torch.tensor(1.))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    [B_, N, C]，B_ = nW * B，N = window_size³
            mask: [nW, N, N] 或 None
                  值为 0（同区域，正常 attend）或 -100（跨区域，禁止 attend）

        Returns:
            x: [B_, N, C]
        """
        B_, N, C = x.shape

        # ── QKV 投影 ──
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B_, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)                    # each: [B_, num_heads, N, head_dim]

        # ── Attention scores ──
        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B_, num_heads, N, N]

        # ── 加入相对位置编码（如启用）──
        if self.use_rel_pos:
            # input_size 在 __init__ 中以 (ws, ws, ws) 传入，这里动态还原
            ws = round(N ** (1 / 3))
            attn_flat = attn.reshape(B_ * self.num_heads, N, N)
            q_flat = q.reshape(B_ * self.num_heads, N, C // self.num_heads)
            attn_flat = add_decomposed_rel_pos(
                attn_flat, q_flat,
                self.rel_pos_h, self.rel_pos_w, self.rel_pos_d,
                (ws, ws, ws), (ws, ws, ws), self.lr
            )
            attn = attn_flat.reshape(B_, self.num_heads, N, N)

        # ── Shifted window mask ──────────────────────────────────────────────
        # mask: [nW, N, N]
        # attn: [B_, num_heads, N, N]，其中 B_ = nW * B
        # 需要将 mask broadcast 到 [B, nW, num_heads, N, N]，再展平为 [B_, num_heads, N, N]
        if mask is not None:
            nW = mask.shape[0]                          # 窗口数
            B = B_ // nW                                # 真实 batch size
            # reshape: [B_, nh, N, N] → [B, nW, nh, N, N]
            attn = attn.view(B, nW, self.num_heads, N, N)
            # mask: [nW, N, N] → [1, nW, 1, N, N]，broadcast 到 [B, nW, nh, N, N]
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            # 展平回 [B_, nh, N, N]
            attn = attn.view(B_, self.num_heads, N, N)
        # ────────────────────────────────────────────────────────────────────

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # ── 加权聚合 + 投影 ──
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, D, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, D, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    pad_d = (window_size - D % window_size) % window_size
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, Dp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Hp, Wp, Dp)


def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int, int], hw: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp, Dp = pad_hw
    H, W, D = hw
    B = windows.shape[0] // (Hp * Wp * Dp // window_size // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, Dp // window_size, window_size, window_size, window_size,
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hp, Wp, Dp, -1)

    if Hp > H or Wp > W or Dp > D:
        x = x[:, :H, :W, :D, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).
    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        rel_pos_d: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
        lr,
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w, q_d = q_size
    k_h, k_w, k_d = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    Rd = get_rel_pos(q_d, k_d, rel_pos_d)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, q_d, dim)
    rel_h = torch.einsum("bhwdc,hkc->bhwdk", r_q, Rh)
    rel_w = torch.einsum("bhwdc,wkc->bhwdk", r_q, Rw)
    rel_d = torch.einsum("bhwdc,dkc->bhwdk", r_q, Rd)

    attn = (
            attn.view(B, q_h, q_w, q_d, k_h, k_w, k_d) +
            lr * rel_h[:, :, :, :, :, None, None] +
            lr * rel_w[:, :, :, :, None, :, None] +
            lr * rel_d[:, :, :, :, None, None, :]
    ).view(B, q_h * q_w * q_d, k_h * k_w * k_d)

    return attn