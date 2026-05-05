from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def _zeros_like(shape: Tuple[int, ...], ref: Tensor) -> Tensor:
    return torch.zeros(*shape, dtype=ref.dtype, device=ref.device)


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, norm: str = 'bn', k: int = 3, p: int = 1):
        super().__init__()
        layers = [nn.Conv2d(c_in, c_out, kernel_size=k, padding=p), nn.GELU()]
        if norm == 'bn':
            layers.append(nn.BatchNorm2d(c_out))
        elif norm == 'gn':
            layers.append(nn.GroupNorm(num_groups=max(1, c_out // 8), num_channels=c_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FDispCNN(nn.Module):

    def __init__(self,
                 k: int,
                 mats_channels: int = 3,
                 pos_channels: int = 2,
                 hidden: int = 64,
                 depth: int = 3,
                 alpha: float = 0.1,
                 norm: str = 'bn'):
        super().__init__()
        assert depth >= 2
        self.k = int(k)
        self.cm = int(mats_channels)
        self.cp = int(pos_channels)
        self.alpha = float(alpha)
        c_in = (self.k + 1) + self.cm + self.cp

        blocks = []
        c_prev = c_in
        for i in range(depth - 1):
            c_next = hidden if i < depth - 2 else hidden
            blocks.append(ConvBlock(c_prev, c_next, norm=norm))
            c_prev = c_next
        self.enc = nn.Sequential(*blocks)
        self.head = nn.Conv2d(c_prev, 1, kernel_size=1)

    def forward(self, E_hist: Tensor, mats: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        # E_hist: [B,k+1,nx,ny]
        B, T, nx, ny = E_hist.shape
        assert T == self.k + 1, f"E_hist length {T} != k+1 ({self.k+1})"

        if mats is None:
            mats = _zeros_like((B, self.cm, nx, ny), E_hist)
        else:
            assert mats.shape[1] == self.cm, f"mats channels {mats.shape[1]} != {self.cm}"
        if pos is None:
            pos = _zeros_like((B, self.cp, nx, ny), E_hist)
        else:
            assert pos.shape[1] == self.cp, f"pos channels {pos.shape[1]} != {self.cp}"

        x = torch.cat([E_hist, mats, pos], dim=1)  # [B, T+Cm+Cp, nx, ny]
        y = self.enc(x)
        dE = self.head(y)
        return self.alpha * torch.tanh(dE)


class FDispTransformer(nn.Module):
    """
    对于每一个(x, y)，取它在时间维度上的一段长度为 (k+1) 的特征序列：[E_{t−k}, ..., E_t]，选择性地拼接上材料特征（mats）
    把这个序列投影到 d_model 然后加上可学习的 time embedding 对应 0..k
    把这个序列输入到带有causal mask的 TransformerEncoder 
    取序列中最后一个时间步的 token 表示，再映射为一个标量输出 ΔE_disp 表示该像素点电场的变化。
    """
    def __init__(self,
                 k: int,
                 mats_channels: int = 3,
                 pos_channels: int = 2,
                 d_model: int = 96,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 192,
                 dropout: float = 0.0,
                 alpha: float = 0.1):
        super().__init__()
        self.k = int(k)
        self.cm = int(mats_channels)
        self.cp = int(pos_channels)
        self.alpha = float(alpha)

        feat_dim = 1 + self.cm + self.cp
        self.proj = nn.Linear(feat_dim, d_model)
        self.time_emb = nn.Embedding(self.k + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=False, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # register a causal mask buffer of maximum length (k+1)
        with torch.no_grad():
            m = torch.full((self.k + 1, self.k + 1), float('-inf'))
            m = torch.triu(m, diagonal=1)  # allow j<=i
        self.register_buffer('causal_mask', m)

    def _prepare_features(self, E_hist: Tensor, mats: Optional[Tensor], pos: Optional[Tensor]) -> Tensor:
        # E_hist: [B,T,nx,ny]; mats: [B,Cm,nx,ny]; pos: [B,Cp,nx,ny]
        B, T, nx, ny = E_hist.shape
        assert T == self.k + 1, f"E_hist length {T} != k+1 ({self.k+1})"

        if mats is None:
            mats = _zeros_like((B, self.cm, nx, ny), E_hist)
        else:
            assert mats.shape[1] == self.cm
        if pos is None:
            pos = _zeros_like((B, self.cp, nx, ny), E_hist)
        else:
            assert pos.shape[1] == self.cp

        # flatten spatial dims -> N = nx*ny sequences per batch
        N = nx * ny
        e = E_hist.view(B, T, N)  # [B,T,N]
        m = mats.view(B, self.cm, N).transpose(1, 2)  # [B,N,Cm]
        p = pos.view(B, self.cp, N).transpose(1, 2)   # [B,N,Cp]

        # replicate mats/pos across time and concat per-token features
        mp = torch.cat([m, p], dim=-1)  # [B,N,Cm+Cp]
        mp_t = mp.unsqueeze(1).expand(B, T, N, self.cm + self.cp)  # [B,T,N,Cm+Cp]
        e_t = e.unsqueeze(-1)  # [B,T,N,1]
        feat = torch.cat([e_t, mp_t], dim=-1)  # [B,T,N,1+Cm+Cp]

        # project to d_model, add time embedding
        feat = self.proj(feat)  # [B,T,N,d_model]
        # add learned time embeddings
        time_ids = torch.arange(T, device=feat.device, dtype=torch.long)
        feat = feat + self.time_emb(time_ids)[None, :, None, :]  # [B,T,N,d_model]

        # to Transformer shape [T, B*N, d_model]
        feat = feat.permute(1, 0, 2, 3).contiguous().view(T, B * N, -1)
        return feat, (B, nx, ny)

    def forward(self, E_hist: Tensor, mats: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        x, (B, nx, ny) = self._prepare_features(E_hist, mats, pos)
        # apply causal mask (size T x T)
        T = x.size(0)
        mask = self.causal_mask[:T, :T]
        z = self.encoder(x, mask=mask)
        z_last = z[-1]  # [B*N, d_model]
        dE = self.head(z_last)  # [B*N,1]
        dE = dE.view(B, 1, nx, ny)
        return self.alpha * torch.tanh(dE)

    @staticmethod
    def make_pos_grid(nx: int, ny: int, device: torch.device, dtype: torch.dtype = torch.float32,
                      include_xy: bool = True, include_idx: bool = False) -> Tensor:
        """
        Returns Tensor [1, C, nx, ny].
        """
        xs = torch.linspace(-1.0, 1.0, steps=nx, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, steps=ny, device=device, dtype=dtype)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        chans = []
        if include_xy:
            chans += [X, Y]
        if include_idx:
            ix = torch.arange(nx, device=device, dtype=dtype).view(nx, 1).expand(nx, ny) / max(1.0, nx - 1)
            iy = torch.arange(ny, device=device, dtype=dtype).view(1, ny).expand(nx, ny) / max(1.0, ny - 1)
            chans += [ix, iy]
        pos = torch.stack(chans, dim=0).unsqueeze(0)  # [1,C,nx,ny]
        return pos


__all__ = [
    'FDispCNN',
    'FDispTransformer',
]
