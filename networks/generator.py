from torch import nn
from .encoder import Encoder
from .styledecoder import Synthesis
import todos
import pdb

class Generator(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)

    # def get_direction(self):
    #     return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def forward(self, img_source, img_drive, h_start=None):
        # todos.debug.output_var("img_source", img_source)
        # todos.debug.output_var("img_drive", img_drive)
        # tensor [img_source] size: [1, 3, 256, 256], min: -1.0, max: 1.0, mean: -0.527514
        # tensor [img_drive] size: [1, 3, 256, 256], min: -0.976471, max: 1.0, mean: -0.140075

        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        img_recon = self.dec(wa, alpha, feats)

        # todos.debug.output_var("img_recon", img_recon)
        # tensor [img_recon] size: [1, 3, 256, 256], min: -1.114883, max: 1.089666, mean: -0.541065

        return img_recon
