import torch
from torch import nn

from .attentions import (
    TextEncoder,
    ResidualCouplingBlock,
    PosteriorEncoder,
    GeneratorNSF,
)


class SynthesizerTrnMsNSFsidM(nn.Module):
    def __init__(
        self,
        spec_channels: int,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
        encoder_dim,
        **kwargs
    ):
        super(SynthesizerTrnMsNSFsidM, self).__init__()
        if isinstance(sr, str):
            sr = {
                "32k": 32000,
                "40k": 40000,
                "48k": 48000,
            }[sr]
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        # self.hop_length = hop_length#
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder(
            encoder_dim,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            is_half=kwargs["is_half"],
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        self.speaker_map = None

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def construct_spkmixmap(self):
        self.speaker_map = torch.zeros((self.n_speaker, 1, 1, self.gin_channels))
        for i in range(self.n_speaker):
            self.speaker_map[i] = self.emb_g(torch.LongTensor([[i]]))
        self.speaker_map = self.speaker_map.unsqueeze(0)

    def forward(self, phone, phone_lengths, pitch, nsff0, g, rnd, max_len=None):
        if self.speaker_map is not None:  # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1)  # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0)  # [B, H, N]
        else:
            g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)

        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * rnd) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], nsff0, g=g)
        return o
