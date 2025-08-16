import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from .text import tokenize, bert_embed, BERT_MODEL_DIM
from torch.utils.data import Dataset, DataLoader
import ipdb
from ..vq_gan_3d.model.vqgan import VQGAN

import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)
class MaskTransformer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, num_heads=4, num_layers=4, patch_size=8, max_seq_len=256):
        """
        Args:
            in_channels (int): 输入 mask 的通道数（例如 7）
            embed_dim (int): Transformer 输出的 embedding 维度，较小可降低内存占用
            num_heads (int): Transformer 多头自注意力头数
            num_layers (int): Transformer 层数
            patch_size (int): 用于 3D Patch 分块的大小（调大 patch_size 可减少 token 数量）
            max_seq_len (int): 预设的位置编码最大长度
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 3D Patch Embedding，将 (B, C, H, W, D) 映射到 (B, embed_dim, H', W', D')
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 展平除批量和通道以外的维度，得到 (B, embed_dim, N)
        self.flatten = nn.Flatten(2)

        # 预设位置编码参数，形状为 (1, embed_dim, max_seq_len)
        self.register_parameter("pos_embedding", nn.Parameter(torch.randn(1, embed_dim, max_seq_len)))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head，将全局 pooling 后的特征进一步映射
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, mask):
        """
        Args:
            mask (Tensor): 输入 multi-channel mask, 形状 (B, C, H, W, D)
        Returns:
            Tensor: 全局 embedding, 形状 (B, embed_dim)
        """
        B = mask.shape[0]
        # 3D Patch Embedding: 输出形状 (B, embed_dim, H', W', D')
        x = self.proj(mask)
        # Flatten 除批量和通道以外的维度: (B, embed_dim, N) 其中 N = H'*W'*D'
        x = self.flatten(x)
        # 调整维度为 Transformer 要求的 (seq_len, B, embed_dim)
        x = x.permute(2, 0, 1)  # (N, B, embed_dim)
        seq_len = x.shape[0]

        # 调整位置编码：如果当前 token 数超过预设长度，则使用插值方式扩展
        if seq_len > self.pos_embedding.shape[2]:
            pos_embed = F.interpolate(self.pos_embedding, size=(seq_len), mode='linear', align_corners=False)
        else:
            pos_embed = self.pos_embedding[:, :, :seq_len]
        # 调整位置编码维度，从 (1, embed_dim, seq_len) -> (seq_len, 1, embed_dim)
        pos_embed = pos_embed.permute(2, 0, 1)

        # 加上位置编码（借助广播，pos_embed 自动扩展到 (seq_len, B, embed_dim)）
        x = x + pos_embed

        # Transformer 编码器
        x = self.transformer(x)  # 输出 (seq_len, B, embed_dim)
        # 对序列进行均值池化，得到 (B, embed_dim)
        x = x.mean(dim=0)
        # 最终通过 MLP head 进一步映射
        embedding = self.mlp_head(x)
        return embedding
# model

class McmaskEmbedding(nn.Module):
    def __init__(self, in_channels=7, out_channels=12, hidden_dim=64, num_w_layers=3):
        """
        Initialize the McmaskEmbedding class based on the described architecture.

        Args:
            in_channels (int): Number of input channels (C) in the multi-channel mask.
            out_channels (int): Number of output channels (E) for the final feature map/embedding.
            hidden_dim (int, optional): Number of channels in hidden layers. Default is 64.
            num_w_layers (int, optional): Number of additional 1x1 conv layers (W₁ to Wₙ). Default is 3.
        """
        super(McmaskEmbedding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        # Two 3x3 convolutional layers
        self.conv1 = nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        # Additional layers for W₁ to Wₙ (implemented as 1x1 convolutions)
        self.w_layers = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(num_w_layers):
            # Gradually reduce channels until reaching out_channels in the last layer
            next_dim = hidden_dim // (2 ** i) if i < num_w_layers - 1 else out_channels
            self.w_layers.append(nn.Conv3d(current_dim, next_dim, kernel_size=1))
            if i < num_w_layers - 1:  # Apply activation except after the final layer
                self.w_layers.append(self.activation)
            current_dim = next_dim

    def forward(self, mask, return_spatial=False):
        """
        Forward pass of the McmaskEmbedding module.

        Args:
            mask (torch.Tensor): Input multi-channel mask of shape (B, C, D, W, H).
            return_spatial (bool, optional): If True, return spatial feature map (B, E, D, W, H).
                                           If False, return global embedding (B, E). Default is True.

        Returns:
            torch.Tensor: Either a spatial feature map (B, E, D, W, H) or an embedding (B, E).
        """
        # Process through two 3x3 convolutions with activation
        x = self.conv1(mask)    # (B, hidden_dim, D, W, H)
        x = self.activation(x)
        x = self.conv2(x)       # (B, hidden_dim, D, W, H)
        x = self.activation(x)

        # Process through W₁ to Wₙ layers
        for layer in self.w_layers:
            x = layer(x)# Final shape: (B, out_channels, D, W, H)

        if return_spatial:
            x = self.softmax(x)  # (B, out_channels, D, W, H)
            return x            # (B, out_channels, D, W, H)
        else:
            # Global average pooling to produce embedding
            embedding = F.adaptive_avg_pool3d(x, 1)  # (B, out_channels, 1, 1, 1)
            embedding = embedding.squeeze(-1).squeeze(-1).squeeze(-1)  # (B, out_channels)
            embedding = self.softmax(embedding)  # (B, out_channels)
            return embedding

# model
class PromptGenBlock(nn.Module):
    # def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 24,lin_dim = 192,label_dim=3):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 16,lin_dim = 192,label_dim=64):
        #label_dim=7 for alltumor label_dim=3 for coronary 12 for maskembedding
        super(PromptGenBlock,self).__init__()
        self.label_dim = label_dim
        self.prompt_len =prompt_len
        self.prompt_dim = prompt_dim
        self.prompt_size =prompt_size
        # self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)

        self.label_encoder = nn.Linear(label_dim, prompt_len * prompt_dim * prompt_size * prompt_size * prompt_size)
        self.conv3x3 = nn.Conv3d(prompt_dim,prompt_dim,kernel_size=(3,3,3),stride=1,padding=1,bias=False)


    def forward(self,x,label):

        B,C,D,H,W = x.shape
        #     device = x.device
        #     random_indices = torch.randint(0, self.label_dim, (B,))
        # # 将类别索引转换为 one-hot 编码
        #     label = torch.zeros(B, self.label_dim).scatter_(1, random_indices.unsqueeze(1), 1)
        #     label = label.to(device)
        #     ipdb.set_trace()
        emb = x.mean(dim=(-3,-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        # ipdb.set_trace()
        label_embedding = self.label_encoder(label).view(B, self.prompt_len, self.prompt_dim, self.prompt_size, self.prompt_size, self.prompt_size)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * label_embedding
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(D,H,W),mode="trilinear")
        # ipdb.set_trace()
        prompt = self.conv3x3(prompt)
        # ipdb.set_trace()

        return prompt
## prompt-lesion
class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))

        # realistically will not be able to generate that many frames of video... yet
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding))

        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)
        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))
        ### multi-channel mask embedding
        # self.McmaskEmbedding = McmaskEmbedding()

        ### multi-channel mask embedding transformer
        self.McmaskEmbedding = MaskTransformer()

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                PromptGenBlock(prompt_dim=dim_out,lin_dim =dim_out),
                block_klass_cond(dim_out * 2, dim_out),
                block_klass_cond(dim_out, dim_out),
                nn.Conv3d(dim_out, dim_out, 1),
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
        #     is_last = ind >= (num_resolutions - 1)
        #
        #     self.ups.append(nn.ModuleList([
        #         block_klass_cond(dim_out * 2, dim_in),
        #         block_klass_cond(dim_in, dim_in),
        #         Residual(PreNorm(dim_in, SpatialLinearAttention(
        #             dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
        #         Residual(PreNorm(dim_in, temporal_attn(dim_in))),
        #         Upsample(dim_in) if not is_last else nn.Identity()
        #     ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
            self,
            x,
            time,
            labels,
            cond=None,
            null_cond_prob=0.,
            focus_present_mask=None,
            # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
            prob_focus_present=0.
    ):
        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        x = torch.cat([x, cond], dim=1)

        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        # ipdb.set_trace()

        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        t = self.time_mlp(time) if exists(self.time_mlp) else None # [2, 128]
        # ipdb.set_trace()

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            ipdb.set_trace()
            cond = torch.where(rearrange(mask, 'b -> b 1'),
                               self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim=-1)
            # ipdb.set_trace()

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        # ipdb.set_trace()

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)
        # x = self.promtGen(x,labels)


        # ipdb.set_trace()

        for promtGen,block1, block2,conv3d,block3, block4, spatial_attn, temporal_attn, upsample in self.ups:

            # ipdb.set_trace()
            # ipdb.set_trace()
            # ## multi-channel mask embedding
            # # ipdb.set_trace()
            embedding = self.McmaskEmbedding(labels)
            # # ipdb.set_trace()
            prompt = promtGen(x,embedding)

            ## label prompt
            # prompt = promtGen(x,labels)
            x = torch.cat((x,prompt),dim =1)
            # ipdb.set_trace()
            x = block1(x,t)
            # ipdb.set_trace()
            x = block2(x,t)
            # ipdb.set_trace()
            x = conv3d(x)
            # ipdb.set_trace()

            x = torch.cat((x, h.pop()), dim=1)
            # ipdb.set_trace()
            x = block3(x, t)
            x = block4(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        # ipdb.set_trace()
        return self.final_conv(x)


    # def forward(
    #     self,
    #     x,
    #     time,
    #     cond=None,
    #     null_cond_prob=0.,
    #     focus_present_mask=None,
    #     # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    #     prob_focus_present=0.
    # ):
    #     assert not (self.has_cond and not exists(cond)
    #                 ), 'cond must be passed in if cond_dim specified'
    #     x = torch.cat([x, cond], dim=1)
    #
    #     batch, device = x.shape[0], x.device
    #
    #     focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
    #         (batch,), prob_focus_present, device=device))
    #
    #     time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
    #     # ipdb.set_trace()
    #
    #     x = self.init_conv(x)
    #     r = x.clone()
    #
    #     x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
    #
    #     t = self.time_mlp(time) if exists(self.time_mlp) else None # [2, 128]
    #     # ipdb.set_trace()
    #
    #     # classifier free guidance
    #
    #     if self.has_cond:
    #         batch, device = x.shape[0], x.device
    #         mask = prob_mask_like((batch,), null_cond_prob, device=device)
    #         ipdb.set_trace()
    #         cond = torch.where(rearrange(mask, 'b -> b 1'),
    #                            self.null_cond_emb, cond)
    #         t = torch.cat((t, cond), dim=-1)
    #         # ipdb.set_trace()
    #
    #     h = []
    #
    #     for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
    #         x = block1(x, t)
    #         x = block2(x, t)
    #         x = spatial_attn(x)
    #         x = temporal_attn(x, pos_bias=time_rel_pos_bias,
    #                           focus_present_mask=focus_present_mask)
    #         h.append(x)
    #         x = downsample(x)
    #
    #     x = self.mid_block1(x, t)
    #     x = self.mid_spatial_attn(x)
    #     x = self.mid_temporal_attn(
    #         x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
    #     x = self.mid_block2(x, t)
    #
    #     for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
    #         x = torch.cat((x, h.pop()), dim=1)
    #         x = block1(x, t)
    #         x = block2(x, t)
    #         x = spatial_attn(x)
    #         x = temporal_attn(x, pos_bias=time_rel_pos_bias,
    #                           focus_present_mask=focus_present_mask)
    #         x = upsample(x)
    #
    #     x = torch.cat((x, r), dim=1)
    #     return self.final_conv(x)


# ## original unet
#
# class Unet3D(nn.Module):
#     def __init__(
#             self,
#             dim,
#             cond_dim=None,
#             out_dim=None,
#             dim_mults=(1, 2, 4, 8),
#             channels=3,
#             attn_heads=8,
#             attn_dim_head=32,
#             use_bert_text_cond=False,
#             init_dim=None,
#             init_kernel_size=7,
#             use_sparse_linear_attn=True,
#             block_type='resnet',
#             resnet_groups=8
#     ):
#         super().__init__()
#         self.channels = channels
#
#         # temporal attention and its relative positional encoding
#
#         rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
#
#         def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
#             dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
#
#         # realistically will not be able to generate that many frames of video... yet
#         self.time_rel_pos_bias = RelativePositionBias(
#             heads=attn_heads, max_distance=32)
#
#         # initial conv
#
#         init_dim = default(init_dim, dim)
#         assert is_odd(init_kernel_size)
#
#         init_padding = init_kernel_size // 2
#         self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
#                                                         init_kernel_size), padding=(0, init_padding, init_padding))
#
#         self.init_temporal_attn = Residual(
#             PreNorm(init_dim, temporal_attn(init_dim)))
#
#         # dimensions
#
#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))
#
#         # time conditioning
#
#         time_dim = dim * 4
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(dim),
#             nn.Linear(dim, time_dim),
#             nn.GELU(),
#             nn.Linear(time_dim, time_dim)
#         )
#
#         # text conditioning
#
#         self.has_cond = exists(cond_dim) or use_bert_text_cond
#         cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim
#
#         self.null_cond_emb = nn.Parameter(
#             torch.randn(1, cond_dim)) if self.has_cond else None
#
#         cond_dim = time_dim + int(cond_dim or 0)
#
#         # layers
#
#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#
#         num_resolutions = len(in_out)
#         # block type
#
#         block_klass = partial(ResnetBlock, groups=resnet_groups)
#         block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)
#
#         # modules for all layers
#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)
#
#             self.downs.append(nn.ModuleList([
#                 block_klass_cond(dim_in, dim_out),
#                 block_klass_cond(dim_out, dim_out),
#                 Residual(PreNorm(dim_out, SpatialLinearAttention(
#                     dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
#                 Residual(PreNorm(dim_out, temporal_attn(dim_out))),
#                 Downsample(dim_out) if not is_last else nn.Identity()
#             ]))
#
#         mid_dim = dims[-1]
#         self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
#
#         spatial_attn = EinopsToAndFrom(
#             'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))
#
#         self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
#         self.mid_temporal_attn = Residual(
#             PreNorm(mid_dim, temporal_attn(mid_dim)))
#
#         self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
#
#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
#             is_last = ind >= (num_resolutions - 1)
#
#             self.ups.append(nn.ModuleList([
#                 block_klass_cond(dim_out * 2, dim_in),
#                 block_klass_cond(dim_in, dim_in),
#                 Residual(PreNorm(dim_in, SpatialLinearAttention(
#                     dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
#                 Residual(PreNorm(dim_in, temporal_attn(dim_in))),
#                 Upsample(dim_in) if not is_last else nn.Identity()
#             ]))
#
#         out_dim = default(out_dim, channels)
#         self.final_conv = nn.Sequential(
#             block_klass(dim * 2, dim),
#             nn.Conv3d(dim, out_dim, 1)
#         )
#
#     def forward_with_cond_scale(
#             self,
#             *args,
#             cond_scale=2.,
#             **kwargs
#     ):
#         logits = self.forward(*args, null_cond_prob=0., **kwargs)
#         if cond_scale == 1 or not self.has_cond:
#             return logits
#
#         null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
#         return null_logits + (logits - null_logits) * cond_scale
#
#     def forward(
#             self,
#             x,
#             time,
#             labels,
#             cond=None,
#             null_cond_prob=0.,
#             focus_present_mask=None,
#             # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
#             prob_focus_present=0.
#     ):
#         assert not (self.has_cond and not exists(cond)
#                     ), 'cond must be passed in if cond_dim specified'
#         x = torch.cat([x, cond], dim=1)
#
#         batch, device = x.shape[0], x.device
#
#         focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
#             (batch,), prob_focus_present, device=device))
#
#         time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
#         # ipdb.set_trace()
#
#         x = self.init_conv(x)
#         r = x.clone()
#
#         x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
#
#         t = self.time_mlp(time) if exists(self.time_mlp) else None # [2, 128]
#         # ipdb.set_trace()
#
#         # classifier free guidance
#
#         if self.has_cond:
#             batch, device = x.shape[0], x.device
#             mask = prob_mask_like((batch,), null_cond_prob, device=device)
#             ipdb.set_trace()
#             cond = torch.where(rearrange(mask, 'b -> b 1'),
#                                self.null_cond_emb, cond)
#             t = torch.cat((t, cond), dim=-1)
#             # ipdb.set_trace()
#
#         h = []
#
#         for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
#             x = block1(x, t)
#             x = block2(x, t)
#             x = spatial_attn(x)
#             x = temporal_attn(x, pos_bias=time_rel_pos_bias,
#                               focus_present_mask=focus_present_mask)
#             h.append(x)
#             x = downsample(x)
#
#         # ipdb.set_trace()
#
#         x = self.mid_block1(x, t)
#         x = self.mid_spatial_attn(x)
#         x = self.mid_temporal_attn(
#             x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
#         x = self.mid_block2(x, t)
#         # ipdb.set_trace()
#
#         for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
#             x = torch.cat((x, h.pop()), dim=1)
#             # ipdb.set_trace()
#             x = block1(x, t)
#             x = block2(x, t)
#             x = spatial_attn(x)
#             x = temporal_attn(x, pos_bias=time_rel_pos_bias,
#                               focus_present_mask=focus_present_mask)
#             x = upsample(x)
#
#         x = torch.cat((x, r), dim=1)
#         return self.final_conv(x)
# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    """
    import math
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    # alphas_cumprod = torch.cos(
    #     ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi  * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False, 
        dynamic_thres_percentile=0.9,
        vqgan_ckpt=None,
        device=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.conv3d = nn.Conv3d(
            in_channels=3,   # 输入通道数，与输入张量的通道数匹配
            out_channels=1,  # 输出通道数，设为1，得到 (b, 1, 96, 96, 96)
            kernel_size=1,   # 核大小设为1，不改变空间尺寸
            stride=1,        # 步幅为1，不改变空间尺寸
            padding=0        # 无填充，不改变空间尺寸
        )
        self.device = device

        if vqgan_ckpt:
            # import ipdb
            # ipdb.set_trace()
            self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
            self.vqgan.eval()
        else:
            self.vqgan = None

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t,labels, clip_denoised: bool, cond=None, cond_scale=1.):
        ## for prompt-lesion
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn.forward_with_cond_scale(x, t,labels, cond=cond, cond_scale=cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.inference_mode()
    @torch.no_grad()
    def p_sample(self, x, t,labels, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, labels=labels,clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.inference_mode()
    @torch.no_grad()
    def p_sample_loop(self, shape,labels, cond=None, cond_scale=1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        print('cond', cond.shape)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long),labels, cond=cond, cond_scale=cond_scale)

        return img

    # @torch.inference_mode()
    @torch.no_grad()
    def sample(self,labels, cond=None, cond_scale=1., batch_size=16):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = batch_size 
        image_size = self.image_size
        channels = 8 # self.channels
        num_frames = self.num_frames
        
        _sample = self.p_sample_loop(
            (batch_size, channels, num_frames, image_size, image_size),labels, cond=cond, cond_scale=cond_scale)

        if isinstance(self.vqgan, VQGAN):
            _sample = (((_sample + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                  self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()

            _sample = self.vqgan.decode(_sample, quantize=True)
        else:
            unnormalize_img(_sample)

        return _sample

    # @torch.inference_mode()
    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, labels,cond=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # ipdb.set_trace()

        if is_list_str(cond):
            cond = bert_embed(
                tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)
        # ipdb.set_trace()


        x_recon = self.denoise_fn(x_noisy, t, labels,cond=cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    # def forward(self, x, *args, **kwargs):
    # def forward(self, img,mask, *args, **kwargs):
    def forward(self, img,mask,label_idx,labels, *args, **kwargs):
        # bs = int(x.shape[0]/2)
        # img=x[:bs,...]
        # mask=x[bs:,...]
        # mask_=(1-mask).detach()
        # masked_img = (img*mask_).detach()
        # masked_img=masked_img.permute(0,1,-1,-3,-2)
        # img=img.permute(0,1,-1,-3,-2)
        # mask=mask.permute(0,1,-1,-3,-2)
        label_idx_ = label_idx
        for i in range(len(label_idx_)):
            if label_idx_[i] ==0:
                label_idx_[i] = 1



        mask_ori = mask.clone() # for maskembdding


        mask = mask[torch.arange(mask.size(0)),label_idx_,:,:,:].unsqueeze(1)

        # mask = mask[:,1,:,:,:].unsqueeze(1)
        mask_=(1-mask).detach()
        # bs = int(x.shape[0]/2)
        # img=x[:bs,...]
        # mask=x[bs:,...]
        # mask_=(1-mask).detach()
        # masked_img = (img*mask_).detach()
        # masked_img=masked_img.permute(0,1,-1,-3,-2)
        # img=img.permute(0,1,-1,-3,-2)
        # mask=mask.permute(0,1,-1,-3,-2)


        # mask_= mask[:,0,:,:,:].unsqueeze(1)
        # ipdb.set_trace()
        masked_img = (img*mask_).detach()
        # ipdb.set_trace()

        masked_img=masked_img.permute(0,1,-1,-3,-2)
        img=img.permute(0,1,-1,-3,-2)
        mask=mask.permute(0,1,-1,-3,-2)

        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                # ipdb.set_trace()
                img = self.vqgan.encode(
                    img, quantize=False, include_embeddings=True)
                # normalize to -1 and 1
                img = ((img - self.vqgan.codebook.embeddings.min()) /
                     (self.vqgan.codebook.embeddings.max() -
                      self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
                
                masked_img = self.vqgan.encode(
                    masked_img, quantize=False, include_embeddings=True)
                # normalize to -1 and 1
                masked_img = ((masked_img - self.vqgan.codebook.embeddings.min()) /
                     (self.vqgan.codebook.embeddings.max() -
                      self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        else:
            print("Hi")
            img = normalize_img(img)
            masked_img = normalize_img(masked_img)
        mask = mask*2.0 - 1.0

        #coronary case
        # ipdb.set_trace()
        # size = (3,masked_img.shape[-3],masked_img.shape[-2],masked_img.shape[-1])
        cc=torch.nn.functional.interpolate(mask, size=masked_img.shape[-3:])
        # cc= self.conv3d(mask)
        # ipdb.set_trace()


        # cc = torch.nn.functional.interpolate(mask, size=masked_img.shape[-3:])
        cond = torch.cat((masked_img, cc), dim=1)

        b, device, img_size, = img.shape[0], img.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # return self.p_losses(img, t, mask_ori,cond=cond, *args, **kwargs)
        return self.p_losses(img, t, labels,cond=cond, *args, **kwargs)

# trainer class

def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5

# trainer clas
from tensorboardX import SummaryWriter
import os
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        cfg,
        folder=None,
        dataset=None,
        *,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=20,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.cfg = cfg
        dl=dataset

        self.len_dataloader = len(dl)
        self.dl = cycle(dl)

        
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(str(self.results_folder)+'/logs'):
            os.makedirs(str(self.results_folder)+'/logs')
        self.writer = SummaryWriter(str(self.results_folder)+'/logs')
        
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'{milestone}.pt'))

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)
        best_train_loss = 0.05
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)
                image = data['image'].cuda()

                ##tumor case
                # mask = data['label'].cuda()
                # mask[mask==1]=0
                # mask[mask==2]=1
                # ipdb.set_trace()


                ##coronary artery case
                mask_ = data['label'].cuda()

                # 获取形状并创建新的形状，将第二个维度设置为 3
                mask_shape = list(mask_.shape)  # 将 torch.Size 转换为列表
                mask_shape[1] = 3  # 修改第二个维度为 3

                # 根据新的形状创建一个新的张量，并移动到 GPU
                mask = torch.zeros(torch.Size(mask_shape)).cuda()

                mask[:,1,:,:,:][mask_.squeeze(1)==3]= 1.0
                mask[:,2,:,:,:][mask_.squeeze(1)==4]= 1.0
                mask[:,0,:,:,:][mask_.squeeze(1)==0]= 1.0
                mask[:,0,:,:,:][mask_.squeeze(1)==1]= 1.0
                mask[:,0,:,:,:][mask_.squeeze(1)==2]= 1.0
                # ipdb.set_trace()




                # input_data = torch.cat([image, mask], dim=0)

                with autocast(enabled=self.amp):
                    # loss = self.model(
                    #     input_data,
                    #     prob_focus_present=prob_focus_present,
                    #     focus_present_mask=focus_present_mask
                    # )
                    loss = self.model(
                        image,
                        mask,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask
                    )

                    self.scaler.scale(
                        loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            lr = self.opt.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('Train_Loss', loss.item(), self.step)
            self.writer.add_scalar('Learning_rate', lr, self.step)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                self.save('model_best')
                print('best model: {} step'.format(self.step // self.save_and_sample_every))

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every

                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')

class Tester(object):
    def __init__(
        self,
        diffusion_model,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema_model = copy.deepcopy(self.model)
        self.step=0
        self.image_size = diffusion_model.image_size

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())


    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)

