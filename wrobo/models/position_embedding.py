import torch
import math
import torch.nn as nn


class SinePositionEmbedding1D(nn.Module):
    """
    Standard sine position encoding for 1D sequences.

    1. If normalize is True:
        x_i = (pos_i / pos_L) * scale
        where pos_L is the maximum index in the sequence.

    2. Positional Encoding calculation:
        PE(pos, 2i)   = sin( pos / temperature^(2i / d_model) )
        PE(pos, 2i+1) = cos( pos / temperature^(2i / d_model) )
    """

    def __init__(self, num_channel, temperature=10000, normalize=False, scale=None):
        super().__init__()
        assert num_channel % 2 == 0, "num_channel should be even. Got {}".format(
            num_channel
        )
        self.num_channel = num_channel
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        Input:
            x: [Batch, Seq_Len, Dim] (Tensor)
        Output:
            pos_encoding: [Batch, Seq_Len, Dim] (Tensor)
        """
        bs, length, _ = x.shape

        ones = torch.ones((bs, length), device=x.device)
        pos_embed = ones.cumsum(1, dtype=torch.float32)  # B, L

        if self.normalize:
            eps = 1e-6
            pos_embed = pos_embed / (pos_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_channel, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_channel)

        pos = pos_embed[:, :, None] / dim_t  # (B, L, 1) / (C,) -> B, L, C
        # [B, L, C/2, 2] -> [B, L, C]
        pos = torch.stack(
            (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        return pos


class SinePositionEmbedding2D(nn.Module):
    """
    Standard sine position encoding for 2D sequences.

    1. Coordinate Map:
        If normalize is True:
            y_i = (pos_y / max_H) * scale
            x_i = (pos_x / max_W) * scale
        Else:
            y_i = pos_y, x_i = pos_x

    2. Frequency Calculation:
        d_f = d_model / 2  (Dimension for each axis)
        omega_k = 1 / temperature^(2k / d_f), where k = i // 2

    3. Component Encoding:
        PE_y(pos_y, 2k)   = sin( y_i * omega_k )
        PE_y(pos_y, 2k+1) = cos( y_i * omega_k )

        PE_x(pos_x, 2k)   = sin( x_i * omega_k )
        PE_x(pos_x, 2k+1) = cos( x_i * omega_k )
    """

    def __init__(self, num_channel, temperature=10000, normalize=False, scale=None):
        super().__init__()
        assert num_channel % 2 == 0, "num_channel should be even. Got {}".format(
            num_channel
        )
        # To match the channel dimension of the input feature map, we divide num_channel by 2
        # for each of the two dimensions (height and width)
        self.num_pos_feats = num_channel // 2
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        Input:
            x: [Batch, C, H, W]
        Output:
            pos_encoding: [Batch, 2 * C, H, W]
        """
        bs, c, h, w = x.shape

        ones = torch.ones((bs, h, w), device=x.device)
        pos_embed_h = ones.cumsum(1, dtype=torch.float32)  # B, H, W
        pos_embed_w = ones.cumsum(2, dtype=torch.float32)  # B, H, W

        if self.normalize:
            eps = 1e-6
            pos_embed_h = pos_embed_h / (pos_embed_h[:, -1:, :] + eps) * self.scale
            pos_embed_w = pos_embed_w / (pos_embed_w[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # [B, H, W, C]
        pos_h = pos_embed_h[:, :, :, None] / dim_t
        pos_w = pos_embed_w[:, :, :, None] / dim_t

        # [B, H, W, C/2, 2] -> [B, H, W, C]
        pos_h = torch.stack(
            (pos_h[:, :, :, 0::2].sin(), pos_h[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_w = torch.stack(
            (pos_w[:, :, :, 0::2].sin(), pos_w[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_h, pos_w), dim=3).permute(0, 3, 1, 2)  # [B, 2*C, H, W]

        return pos


class LearnablePositionEmbedding1D(nn.Module):
    """
    Absolute learnable 1D pos embedding

    Mathematical Form:
    PE(pos) = EmbeddingTable[pos]
    where EmbeddingTable is a trainable parameter of shape [max_len, num_channel].
    """

    def __init__(self, num_channel, max_len=5000):
        super().__init__()
        # A learnable embedding table
        self.pe = nn.Embedding(max_len, num_channel)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pe.weight)

    def forward(self, x):
        """
        x: [B, Seq_Len, Dim]
        """
        bs, length, _ = x.shape

        # position idx [0, 1, 2, ..., seq_len-1]
        position = torch.arange(length, device=x.device).unsqueeze(0)  # [1, L]

        # map position idx to embedding [B, L, D]
        return self.pe(position).expand(bs, -1, -1)


class LearnablePositionEmbedding2D(nn.Module):
    """
    Learnable position encoding for 2D images.

    Logic:
    1. Define separate embedding tables for Row (H) and Column (W).
    2. Final PE = Concat(Row_Embed, Col_Embed)
    """

    def __init__(self, num_channel, max_h=500, max_w=500):
        super().__init__()
        assert num_channel % 2 == 0, "num_channel must be even for 2D concatenation"
        num_pos_feats = num_channel // 2
        self.h_embed = nn.Embedding(max_h, num_pos_feats)
        self.w_embed = nn.Embedding(max_w, num_pos_feats)

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.h_embed.weight)
        nn.init.uniform_(self.w_embed.weight)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        bs, c, h, w = x.shape

        # position idx for row and column, [H,]/[W,]
        i = torch.arange(h, device=x.device)
        j = torch.arange(w, device=x.device)

        # map idx to embed
        h_emb = (
            self.h_embed(i).unsqueeze(1).repeat(1, w, 1)
        )  # [H, 1, C/2] -> [H, W, C/2]
        w_emb = (
            self.w_embed(j).unsqueeze(0).repeat(h, 1, 1)
        )  # [1, W, C/2] -> [H, W, C/2]

        # [H, W, C] -> [C, H, W] -> [B, C, H, W]
        pos = (
            torch.cat([h_emb, w_emb], dim=-1)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(bs, -1, -1, -1)
        )

        return pos  # [B, C, H, W]
