import torchvision

from torch import nn

from wrobo.methods.ACT.models.backbone import ResNetBackBone
from wrobo.methods.ACT.models.transformer import DETRVAE
from wrobo.models.position_embedding import SinePositionEmbedding2D


class ACTPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        backbone = ResNetBackBone(
            backbone_name=config["backbone_name"],
            backbone_weights_name=config["backbone_weights_name"],
        )
        self.pe = SinePositionEmbedding2D(self.config["embed_dim"])
        self.model = DETRVAE(backbone=backbone, backbone_PE=self.pe,  **self.get_model_settings())

    def forward(self, image, proprio_state, actions=None, is_pad=None):
        if actions is not None:  # Training time
            actions = actions[:, : self.model.action_chunk_size]
            is_pad = is_pad[:, : self.model.action_chunk_size]
            a_hat, _, (mu, logvar) = self.model(
                image=image,
                proprio_state=proprio_state,
                env_state=None,
                actions=actions,
                is_pad=is_pad,
            )
            return {"action_chunk": a_hat, "mu": mu, "logvar": logvar}
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                image=image, proprio_state=proprio_state, env_state=None
            )  # no action, sample from prior
            return {"action_chunk": a_hat}

    def get_model_settings(self):
        return {
            "history_width": self.config["history_width"],
            "action_chunk_size": self.config["action_chunk_size"],
            "proprio_dim": self.config["proprio_dim"],
            "env_dim": self.config["env_dim"],
            "z_dim": self.config["z_dim"],
            "embed_dim": self.config["embed_dim"],
            "nhead": self.config["nhead"],
            "num_encoder_layers": self.config["num_encoder_layers"],
            "num_decoder_layers": self.config["num_decoder_layers"],
            "FFN_dim": self.config["FFN_dim"],
            "dropout": self.config["dropout"],
            "activation": self.config["activation"],
            "pre_norm": self.config["pre_norm"],
            "return_intermediate_decoder": self.config["deep_supervision"],
        }
