import torch.nn as nn
from models.MS_RadarFormer.Encoder import Encoder
from models.MS_RadarFormer.Decoder import Decoder
from models.MS_RadarFormer.model_utils import PatchEmbedBack3D, ConvOut
from einops import rearrange


class FusionFormer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.encoder = Encoder(
            configs=configs,
            patch_size=configs["model_patch_size"],
            in_chans=configs["img_channel"] * (configs["patch_size"] ** 2),
            embed_dim=configs["embed_dim"],
            depths=configs["depths"],
            num_heads=configs["num_heads"],
            window_size=configs["window_size"],
            drop_rate=configs["drop_rate"],
            attn_drop_rate=configs["attn_drop_rate"],
            drop_path_rate=configs["drop_path_rate"],
        )
        self.decoder = Decoder(
            configs=configs,
            patch_size=configs["model_patch_size"],
            in_chans=configs["img_channel"] * (configs["patch_size"] ** 2),
            embed_dim=configs["embed_dim"],
            depths=configs["depths"],
            num_heads=configs["num_heads"],
            window_size=configs["window_size"],
            drop_rate=configs["drop_rate"],
            attn_drop_rate=configs["attn_drop_rate"],
            drop_path_rate=configs["drop_path_rate"],
        )
        self.patch_embed_back = PatchEmbedBack3D(
            patch_size=configs["model_patch_size"],
            # in_chans为输出channel
            in_chans=configs["embed_dim"]
            if configs["use_multi_resolution_branch"] == 1
            else configs["embed_dim"] // 2,
            embed_dim=configs["embed_dim"] * 2  # embed_dim为输入channel
            if configs["use_multi_resolution_branch"] == 1
            else configs["embed_dim"],
        )
        self.conv_out = ConvOut(
            configs,
            configs["embed_dim"]
            if configs["use_multi_resolution_branch"] == 1
            else configs["embed_dim"] // 2,
            configs["img_out_channel"] * (configs["patch_size"] ** 2),
        )
        # 添加ReLU激活函数，确保输出非负
        self.relu = nn.ReLU()

    def forward(self, x):
        x = rearrange(x, "b t c h w -> b c t h w")
        memory, memory_low_res = self.encoder(x)
        out = self.decoder(memory, memory_low_res)
        out = self.patch_embed_back(out)
        out = self.conv_out(out)
        # 应用ReLU确保输出非负
        out = self.relu(out)
        out = rearrange(out, "b c t h w -> b t c h w")
        return out
