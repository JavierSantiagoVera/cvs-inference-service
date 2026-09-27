"""Perceiver temporal con compuertas de PercEVA-CVS.

Adaptado de models/TemporalPerceiver_concat_gate.py en
https://github.com/BCV-Uniandes/PercEVA-CVS (CC BY-NC-SA 4.0), de Cañar, Vera,
Tovar y Arbeláez. Solo conserva la configuración publicada para SAGES 2024:
posiciones aprendidas, compuertas en los residuales, lectura desde el keyframe
y una sola cabeza. Los nombres de los módulos no cambian, de modo que los
checkpoints originales cargan con strict=True.
"""

from __future__ import annotations

import torch
from torch import nn


class PerceiverLiteTemporalGated(nn.Module):
    """Perceiver cuyos latentes, más el token del keyframe, atienden a la ventana."""

    def __init__(
        self,
        d_in: int = 1024,
        d_model: int = 1024,
        nhead: int = 16,
        num_layers: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.0,
        n_classes: int = 3,
        K: int = 64,  # noqa: N803 - nombre del config original
        mlp_hidden: int = 128,
        pe_max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Identity() if d_in == d_model else nn.Linear(d_in, d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.pos_enc = nn.Embedding(pe_max_len, d_model)
        self.latents = nn.Parameter(torch.randn(K, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(d_model),
                        "cross": nn.MultiheadAttention(
                            d_model, nhead, dropout=dropout, batch_first=True
                        ),
                        "ln2": nn.LayerNorm(d_model),
                        "self": nn.MultiheadAttention(
                            d_model, nhead, dropout=dropout, batch_first=True
                        ),
                        "ln3": nn.LayerNorm(d_model),
                        "ff": nn.Sequential(
                            nn.Linear(d_model, dim_ff),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim_ff, d_model),
                            nn.Dropout(dropout),
                        ),
                        "gate_cross": nn.Linear(d_model, d_model),
                        "gate_self": nn.Linear(d_model, d_model),
                        "gate_ff": nn.Linear(d_model, d_model),
                    }
                )
            )

        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_classes),
        )

    def forward(
        self, x: torch.Tensor, key_index: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Calcula los logits [B, n_classes].

        Args:
            x: embeddings de la ventana, [B, T, d_in]. Los huecos van en cero; el
                modelo original no usa máscara y atiende también a ellos.
            key_index: posición del frame a predecir dentro de la ventana, [B].
            positions: índice temporal absoluto de cada posición, [B, T].
        """
        h = self.in_norm(self.in_proj(x))
        h = h + self.pos_enc(positions).to(h.dtype)

        batch = torch.arange(x.shape[0], device=x.device)
        key_feat = h[batch, key_index].unsqueeze(1)
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        lat = torch.cat([key_feat, latents], dim=1)

        for block in self.layers:
            q = block["ln1"](lat)
            cross, _ = block["cross"](q, h, h, need_weights=False)
            lat = torch.sigmoid(block["gate_cross"](q)) * cross + lat

            q = block["ln2"](lat)
            self_out, _ = block["self"](q, q, q, need_weights=False)
            lat = torch.sigmoid(block["gate_self"](q)) * self_out + lat

            q = block["ln3"](lat)
            lat = torch.sigmoid(block["gate_ff"](q)) * block["ff"](q) + lat

        return self.head(self.out_norm(lat[:, 0]))
