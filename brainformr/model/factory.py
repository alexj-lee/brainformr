from typing import Optional

from torch import nn


def _init_xformer_weights(m: nn.Module):
    # see: unsure if it has been fixed as of 2023-12-04 https://github.com/pytorch/pytorch/issues/72253
    for param_name, param in m.named_parameters():
        if (
            "decoder" in param_name
            and ("linear" in param_name or "proj" in param_name)
            and "bias" not in param_name
        ):
            nn.init.kaiming_normal_(param)

        if (
            "encoder" in param_name
            and ("linear" in param_name or "proj" in param_name)
            and "bias" not in param_name
        ):
            nn.init.kaiming_normal_(param)


def get_projection_layers(n_genes: int, feature_dim: int) -> nn.Sequential:
    modules = nn.Sequential(
        nn.Linear(n_genes, feature_dim),
        nn.LayerNorm(feature_dim),
        nn.GELU(),
        nn.Linear(feature_dim, feature_dim),
        nn.LayerNorm(feature_dim),
    )
    return modules


def set_up_transformer_layers(
    embedding_dim: int,
    num_heads: int,
    depth: int,
    dropout_p: Optional[float] = 0.0,
    bias: Optional[bool] = True,
    zero_attn: Optional[bool] = True,
):
    layer = nn.TransformerEncoderLayer(
        d_model=embedding_dim,
        nhead=num_heads,
        activation="gelu",
        dropout=dropout_p,
    )

    layer.self_attn = nn.MultiheadAttention(
        embed_dim=embedding_dim,
        num_heads=num_heads,
        dropout=dropout_p,
        add_bias_kv=bias,
        add_zero_attn=zero_attn,
    )

    xformer_blocks = nn.TransformerEncoder(
        encoder_layer=layer, num_layers=depth, enable_nested_tensor=False
    )

    _init_xformer_weights(xformer_blocks)

    return xformer_blocks


def _init_xformer_weights(m: nn.Module):
    # see: unsure if it has been fixed as of 2023-12-04 https://github.com/pytorch/pytorch/issues/72253
    for param_name, param in m.named_parameters():
        if (
            "linear" in param_name or "proj" in param_name
        ) and "bias" not in param_name:
            nn.init.kaiming_normal_(param)

        if (
            "linear" in param_name or "proj" in param_name
        ) and "bias" not in param_name:
            nn.init.kaiming_normal_(param)
