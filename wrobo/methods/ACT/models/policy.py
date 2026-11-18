from torch import nn

from wrobo.methods.ACT.models.backbone import ResNetBackBone
from wrobo.methods.ACT.models.dimensions import resolve_act_dimensions
from wrobo.methods.ACT.models.transformer import DETRVAE
from wrobo.methods.ACT.models.position_embedding import SinePositionEmbedding2D


class ACTPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proprio_dim, self.action_dim = resolve_act_dimensions(
            config.get("proprio_dim"), config.get("action_dim")
        )

        backbone = ResNetBackBone(
            backbone_name=config["backbone_name"],
            backbone_weights_name=config["backbone_weights_name"],
            freeze_bn=config.get("freeze_bn", True),  # Default to True if not specified
            load_pretrained=config.get("backbone_load_pretrained", True),
        )
        self.pe = SinePositionEmbedding2D(self.config["embed_dim"])
        self.model = DETRVAE(
            backbone=backbone, backbone_PE=self.pe, **self.get_model_settings()
        )

    def forward(self, image, proprio_state, actions=None, is_pad=None, camera_names=None):
        if actions is not None:  # Training time
            actions = actions[:, : self.model.action_chunk_size]
            is_pad = is_pad[:, : self.model.action_chunk_size]
            a_hat, _, (mu, logvar) = self.model(
                image=image,
                proprio_state=proprio_state,
                env_state=None,
                actions=actions,
                is_pad=is_pad,
                camera_names=camera_names,
            )
            return {
                "action_chunk": a_hat,
                "mu": mu,
                "logvar": logvar,
                "pred": a_hat[0] if isinstance(a_hat, list) else a_hat,
            }
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                image=image,
                proprio_state=proprio_state,
                env_state=None,
                camera_names=camera_names,
            )  # no action, sample from prior
            return {
                "action_chunk": a_hat,
                "pred": a_hat[0] if isinstance(a_hat, list) else a_hat,
            }

    def get_model_settings(self):
        return {
            "history_width": self.config["history_width"],
            "camera_names": self.config.get("camera_names", None),
            "action_chunk_size": self.config["action_chunk_size"],
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
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
