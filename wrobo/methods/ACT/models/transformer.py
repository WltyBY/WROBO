import torch

import numpy as np

from typing import Optional
from torch import Tensor, nn

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        nhead,
        FFN_dim=2048,
        dropout=0.1,
        activation="relu",
        pre_norm=True,
    ):
        super().__init__()

        # MHSA
        self.self_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True
        )
        self.dropout_SA = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # FFN
        self.linear1 = nn.Linear(embed_dim, FFN_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(FFN_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = ACTIVATION_FUNCTIONS[activation]()

        self.pre_norm = pre_norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        x,
        x_mask: Optional[Tensor] = None,
        x_key_padding_mask: Optional[Tensor] = None,
        x_pos: Optional[Tensor] = None,
    ):
        # MHSA
        x_norm = self.norm1(x) if self.pre_norm else x

        q = k = self.with_pos_embed(x_norm, x_pos)
        x_attn = self.self_attn(
            q, k, value=x_norm, attn_mask=x_mask, key_padding_mask=x_key_padding_mask
        )[0]
        x = x + self.dropout_SA(x_attn)

        x = x if self.pre_norm else self.norm1(x)

        # FFN
        x_norm = self.norm2(x) if self.pre_norm else x

        x_ffn = self.linear2(self.dropout1(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(x_ffn)

        x = x if self.pre_norm else self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        nhead,
        FFN_dim=2048,
        dropout=0.1,
        activation="relu",
        pre_norm=True,
    ):
        super().__init__()

        # MHSA
        self.self_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True
        )
        self.dropout_SA = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # MHCA
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True
        )
        self.dropout_CA = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN
        self.linear1 = nn.Linear(embed_dim, FFN_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(FFN_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.activation = ACTIVATION_FUNCTIONS[activation]()

        self.pre_norm = pre_norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        x,
        context,
        x_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        x_key_padding_mask: Optional[Tensor] = None,
        context_key_padding_mask: Optional[Tensor] = None,
        x_pos: Optional[Tensor] = None,
        context_pos: Optional[Tensor] = None,
    ):
        # MHSA
        x_norm = self.norm1(x) if self.pre_norm else x

        q = k = self.with_pos_embed(x_norm, x_pos)
        x_attn = self.self_attn(
            q, k, value=x_norm, attn_mask=x_mask, key_padding_mask=x_key_padding_mask
        )[0]
        x = x + self.dropout_SA(x_attn)

        x = x if self.pre_norm else self.norm1(x)

        # MHCA
        x_norm = self.norm2(x) if self.pre_norm else x
        # context will be normalized only when pre_norm is True.
        # And it has been normalized in Encoder.

        x_attn = self.cross_attn(
            query=self.with_pos_embed(x_norm, x_pos),
            key=self.with_pos_embed(context, context_pos),
            value=context,
            attn_mask=context_mask,
            key_padding_mask=context_key_padding_mask,
        )[0]
        x = x + self.dropout_CA(x_attn)

        x = x if self.pre_norm else self.norm2(x)

        # FFN
        x_norm = self.norm3(x) if self.pre_norm else x

        x_ffn = self.linear2(self.dropout1(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(x_ffn)

        x = x if self.pre_norm else self.norm3(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        nhead,
        FFN_dim=2048,
        dropout=0.1,
        activation="relu",
        pre_norm=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    nhead=nhead,
                    FFN_dim=FFN_dim,
                    dropout=dropout,
                    activation=activation,
                    pre_norm=pre_norm,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.pre_norm = pre_norm
        if self.pre_norm:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x,
        x_mask: Optional[Tensor] = None,
        x_key_padding_mask: Optional[Tensor] = None,
        x_pos: Optional[Tensor] = None,
    ):
        for layer in self.layers:
            x = layer(
                x,
                x_mask=x_mask,
                x_key_padding_mask=x_key_padding_mask,
                x_pos=x_pos,
            )

        # normalize only when pre_norm is True for the features here is context for decoder.
        # it need to be normalized here.
        if self.pre_norm:
            x = self.norm(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        nhead,
        FFN_dim=2048,
        dropout=0.1,
        activation="relu",
        pre_norm=True,
        return_intermediate=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embed_dim=embed_dim,
                    nhead=nhead,
                    FFN_dim=FFN_dim,
                    dropout=dropout,
                    activation=activation,
                    pre_norm=pre_norm,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.return_intermediate = return_intermediate

    def forward(
        self,
        x,
        context,
        x_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        x_key_padding_mask: Optional[Tensor] = None,
        context_key_padding_mask: Optional[Tensor] = None,
        x_pos: Optional[Tensor] = None,
        context_pos: Optional[Tensor] = None,
    ):
        intermediate = []

        for layer in self.layers:
            x = layer(
                x,
                context,
                x_mask=x_mask,
                context_mask=context_mask,
                x_key_padding_mask=x_key_padding_mask,
                context_key_padding_mask=context_key_padding_mask,
                x_pos=x_pos,
                context_pos=context_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(x))

        x = self.norm(x)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(x)
            # num_layers, batch_size, seq, embed_dim
            return intermediate[::-1]  # from the deeper layer to the former layer

        # 1, batch_size, seq, embed_dim
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        FFN_dim=2048,
        dropout=0.1,
        activation="relu",
        pre_norm=False,
        return_intermediate_decoder=False,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            embed_dim=embed_dim,
            nhead=nhead,
            FFN_dim=FFN_dim,
            dropout=dropout,
            activation=activation,
            pre_norm=pre_norm,
        )

        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            embed_dim=embed_dim,
            nhead=nhead,
            FFN_dim=FFN_dim,
            dropout=dropout,
            activation=activation,
            pre_norm=pre_norm,
            return_intermediate=return_intermediate_decoder,
        )

        self._reset_parameters()

        self.embed_dim = embed_dim
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x,
        mask,
        q_pos_embed,
        c_pos_embed,
    ):

        tgt = torch.zeros_like(q_pos_embed)
        context = self.encoder(x, x_key_padding_mask=mask, x_pos=c_pos_embed)
        hs = self.decoder(
            tgt,
            context,
            context_key_padding_mask=mask,
            x_pos=q_pos_embed,
            context_pos=c_pos_embed,
        )
        # hs: (num_layers, batch_size, seq_de, embed_dim)
        # context: (1, batch_size, seq_en, embed_dim)
        return hs, context.unsqueeze(0)


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_PE,
        history_width,
        action_chunk_size=4,
        proprio_dim=14,
        env_dim=7,
        z_dim=32,
        embed_dim=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        FFN_dim=2048,
        dropout=0.1,
        activation="relu",
        pre_norm=False,
        return_intermediate_decoder=False,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.env_dim = env_dim
        self.z_dim = z_dim
        self.action_chunk_size = action_chunk_size

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            embed_dim=embed_dim,
            nhead=nhead,
            FFN_dim=FFN_dim,
            dropout=dropout,
            activation=activation,
            pre_norm=pre_norm,
        )

        self.transformer = Transformer(
            embed_dim=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            FFN_dim=FFN_dim,
            dropout=dropout,
            activation=activation,
            pre_norm=pre_norm,
            return_intermediate_decoder=return_intermediate_decoder,
        )

        self.time_embed = nn.Embedding(max(history_width, 50), embed_dim)
        if history_width == 1:
            self.time_embed.weight.requires_grad_(False)
            with torch.no_grad():
                self.time_embed.weight.zero_()

        # encoder extra parameters
        self.encoder_action_proj = nn.Linear(
            self.proprio_dim, embed_dim
        )  # project action to embedding
        self.encoder_proprio_proj = nn.Linear(
            self.proprio_dim, embed_dim
        )  # project qpos to embedding
        self.cls_token = nn.Embedding(1, embed_dim)  # extra cls token embedding
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + action_chunk_size, embed_dim),
        )  # Seq: [CLS], joint, action_seq, Out: (1, 1+1+action_chunk_size, embed_dim)
        self.z_proj = nn.Linear(
            embed_dim, self.z_dim * 2
        )  # project hidden state to latent std, var, (bs, z_dim*2)

        # extra decoder parameters
        self.z_reproj = nn.Linear(
            self.z_dim, embed_dim
        )  # project latent sample z to embedding
        if backbone is not None:
            self.backbone = backbone
            self.backbone_PE = backbone_PE
            self.img_proj = nn.Conv2d(backbone.num_channels, embed_dim, kernel_size=1)
            self.decoder_proprio_proj = nn.Linear(self.proprio_dim, embed_dim)
        else:
            self.backbone = None
            self.backbone_PE = None
            self.env_proj = nn.Linear(self.env_dim, embed_dim)
            self.decoder_proprio_proj = nn.Linear(self.proprio_dim, embed_dim)
            self.pos_env = nn.Embedding(1, embed_dim)
        # learned position embedding for proprio and latent
        self.proprio_z_pos_embed = nn.Embedding(2, embed_dim)
        self.action_chunk_embed = nn.Embedding(action_chunk_size, embed_dim)
        self.action_head = nn.Linear(embed_dim, proprio_dim)
        self.is_pad_head = nn.Linear(embed_dim, 1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, image, proprio_state, env_state, actions=None, is_pad=None):
        """
        image: batch, num_cam, his_width, channel, height, width
        proprio_state: batch, his_width, proprio_dim
        env_state: None
        actions: batch, seq, action_dim
        """
        bs, num_cam, his_width, *_ = image.shape
        ################# VAE Encoder forward #################
        # Prepare style variable z
        if actions is not None:  # Training
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, embed_dim)
            proprio_embed = self.encoder_proprio_proj(
                proprio_state
            )  # (bs, his_width, embed_dim)

            cls_embed = self.cls_token.weight  # (1, embed_dim)
            cls_embed = cls_embed.unsqueeze(0).repeat(bs, 1, 1)  # (bs, 1, embed_dim)
            encoder_input = torch.cat(
                [cls_embed, proprio_embed, action_embed], axis=1
            )  # (bs, 1+his_width+seq, embed_dim)

            cls_joint_is_pad = torch.full((bs, 1 + his_width), False).to(
                proprio_state.device
            )  # Don't ignore cls token and proprio state
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)

            # obtain position embedding
            # add time_embedding for proprioception embedding
            proprio_pos_embed_wo_time = self.pos_table[:, 1][
                :, None
            ]  # (1, 1, embed_dim)
            t_embed = self.time_embed.weight[:his_width][
                None
            ]  # (1, his_width, embed_dim)
            proprio_pos_embed_w_time = (
                proprio_pos_embed_wo_time + t_embed
            )  # (1, his_width, embed_dim)
            pos_embed_w_time = torch.cat(
                [
                    self.pos_table[:, 0][:, None],  # (1, 1, embed_dim)
                    proprio_pos_embed_w_time,
                    self.pos_table[:, 2:],
                ],
                axis=1,
            ).repeat(
                bs, 1, 1
            )  # bs, his_width+1, embed_dim

            # query model
            encoder_output = self.encoder(
                encoder_input, x_pos=pos_embed_w_time, x_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[:, 0]  # take cls output only
            # get style variable z
            z_info = self.z_proj(encoder_output)
            mu = z_info[:, : self.z_dim]
            logvar = z_info[:, self.z_dim :]
            z_sample = self.reparametrize(mu, logvar)
        else:  # Inference
            mu = logvar = None
            z_sample = torch.zeros([bs, self.z_dim], dtype=torch.float32).to(
                proprio_state.device
            )

        ################# VAE Decoder forward #################
        # Get style variable output embedding
        z = self.z_reproj(z_sample).unsqueeze(1)  # (bs, 1, embed_dim)

        if self.backbone is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id in range(num_cam):
                # (bs, his_width, channel, height, width)
                cam_imgs = image[:, cam_id]
                cam_imgs = cam_imgs.flatten(
                    0, 1
                )  # (bs*his_width, channel, height, width)

                features = self.backbone(cam_imgs)[
                    "4"
                ]  # take the last layer (layer_4) feature
                pos = self.backbone_PE(
                    features
                )  # (bs*his_width, embed_dim, height', width')

                features = self.img_proj(
                    features
                )  # (bs*his_width, embed_dim, height', width')

                features = features.view(
                    bs, his_width, *features.shape[1:]
                )  # (bs, his_width, embed_dim, height', width')
                pos = pos.view(
                    bs, his_width, *pos.shape[1:]
                )  # (bs, his_width, embed_dim, height', width')

                # Time/History embedding
                t_embed = self.time_embed.weight[:his_width][
                    None, :, :, None, None
                ]  # (1, his_width, embed_dim)  # broadcast, (1, T, D, 1, 1)

                pos = pos + t_embed

                all_cam_features.append(features.flatten(3))  # (bs, his_width, embed_dim, height'*width')
                all_cam_pos.append(pos.flatten(3))  # (bs, his_width, embed_dim, height'*width')

            # fold camera dimension into width dimension
            # all_cam_features: [(bs, his_width*H*W, embed_dim), ...] -> (bs, his_width, num_cam, embed_dim, H*W)
            cam_features = torch.stack(all_cam_features, axis=2).transpose(-1, -2).flatten(1, 3)  # (bs, his_width*num_cam*H*W, embed_dim)
            cam_pos = torch.stack(all_cam_pos, axis=2).transpose(-1, -2).flatten(1, 3)  # (bs, his_width*num_cam*H*W, embed_dim)

            # (action_chunk_size, embed_dim) -> (bs, action_chunk_size, embed_dim)
            AC_pos_embed = self.action_chunk_embed.weight[None].repeat(bs, 1, 1)

            # proprioception features
            proprio_input = self.decoder_proprio_proj(
                proprio_state
            )  # (bs, his_width, embed_dim)
            # proprioception and z pos_embed
            proprio_pos_embed = self.proprio_z_pos_embed.weight[0][
                None, None, :
            ]  # (1, 1, embed_dim)
            t_embed = self.time_embed.weight[:his_width][
                None
            ]  # (1, his_width, embed_dim)
            proprio_pos_embed = proprio_pos_embed + t_embed  # (1, his_width, embed_dim)
            proprio_z_pos_embed = torch.cat(
                [proprio_pos_embed, self.proprio_z_pos_embed.weight[1][None, None, :]],
                axis=1,
            ).repeat(
                bs, 1, 1
            )  # bs, his_width+1, embed_dim

            # formulate input of tranformer
            input = torch.cat([cam_features, proprio_input, z], axis=1)
            pos = torch.cat([cam_pos, proprio_z_pos_embed], axis=1)

            hs = self.transformer(
                x=input,
                mask=None,
                q_pos_embed=AC_pos_embed,
                c_pos_embed=pos,
            )[0]
        else:
            env_state = self.env_proj(env_state)  # (bs, his_width, embed_dim)
            proprio_state = self.decoder_proprio_proj(
                proprio_state
            )  # (bs, his_width, embed_dim)
            transformer_input = torch.cat(
                [env_state, proprio_state, z], axis=1
            )  # (bs, his_width+his_width, embed_dim)

            proprio_pos_embed = self.proprio_z_pos_embed.weight[0][
                None, None
            ]  # (1,1,embed_dim)
            t_embed = self.time_embed.weight[:his_width][
                None
            ]  # (1, his_width, embed_dim)
            env_pos = self.pos_env.weight[None] + t_embed  # (1, his_width, embed_dim)
            proprio_pos_embed = proprio_pos_embed + t_embed  # (1, his_width, embed_dim)
            pos = torch.cat(
                [
                    env_pos,
                    proprio_pos_embed,
                    self.proprio_z_pos_embed.weight[1][None, None, :],
                ],
                axis=1,
            ).repeat(
                bs, 1, 1
            )  # (bs, his_width+1, embed_dim)

            hs = self.transformer(
                x=transformer_input,
                mask=None,
                q_pos_embed=self.action_chunk_embed.weight[None].repeat(bs, 1, 1),
                c_pos_embed=pos,
            )[0]

        # (num_layers, batch_size, seq_de, embed_dim) -> (num_layers, batch_size, seq_de, proprio_dim)
        a_hat = [self.action_head(h) for h in hs]
        is_pad_hat = [self.is_pad_head(h) for h in hs]
        return a_hat, is_pad_hat, [mu, logvar]
