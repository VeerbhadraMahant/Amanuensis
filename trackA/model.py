"""A small CTC acoustic model: strided conv front end, then either transformer encoder layers or a bidirectional GRU.

Both are small enough to train from scratch on an 8 GB GPU. The GRU is the default: on ~10 h of speech the
transformer sat on the CTC plateau (predicting character frequencies) for most of a 30-epoch budget, while recurrent
encoders are known to converge much faster at this data scale.
"""
import math

import torch
from torch import nn

from trackA.text import VOCAB


def out_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Frames after two stride-2 convs (kernel 3, padding 1): ceil(T / 2) twice."""
    return ((lengths + 1) // 2 + 1) // 2


class CtcModel(nn.Module):
    def __init__(self, n_mels: int = 80, vocab: int = len(VOCAB), d_model: int = 256, layers: int = 4, heads: int = 4, dropout: float = 0.1,
                 arch: str = "gru"):
        super().__init__()
        if arch not in ("gru", "transformer"):
            raise ValueError("arch must be 'gru' or 'transformer'")
        self.arch = arch
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        freq = ((n_mels + 1) // 2 + 1) // 2
        self.proj = nn.Linear(32 * freq, d_model)
        self.dropout = nn.Dropout(dropout)
        if arch == "transformer":
            enc = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        else:  # each direction has d_model // 2 units so the output width is d_model
            self.encoder = nn.GRU(d_model, d_model // 2, num_layers=layers, batch_first=True, bidirectional=True,
                                  dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(d_model, vocab)
        self.d_model = d_model

    def _positions(self, t: int, device) -> torch.Tensor:
        pos = torch.arange(t, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, device=device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(t, self.d_model, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats (B, T, n_mels), lengths (B,) -> log-probs (B, T', vocab), output lengths (B,)."""
        x = self.conv(feats.unsqueeze(1))  # (B, C, T', F')
        b, c, t, f = x.shape
        x = self.proj(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
        out_lens = out_lengths(lengths)
        if self.arch == "transformer":
            x = self.dropout(x + self._positions(t, x.device))
            pad_mask = torch.arange(t, device=x.device)[None, :] >= out_lens[:, None]
            x = self.encoder(x, src_key_padding_mask=pad_mask)
        else:
            # Packing keeps padding out of the backward direction, so a sequence's output does not depend on its batch.
            packed = nn.utils.rnn.pack_padded_sequence(self.dropout(x), out_lens.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = nn.utils.rnn.pad_packed_sequence(self.encoder(packed)[0], batch_first=True, total_length=t)
        return self.out(x).log_softmax(-1), out_lens
