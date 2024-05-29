import torch
from scvi import distributions
from torch import nn

from brainformr import model

# some kind of ad-hoc test for tolerance on some known output

def test_xformer_factory_pytorch():
    embed_dim = 64
    depth = 3
    num_heads = 8
    dropout = 0.0
    zero_attn = True

    for bias in (True, False):
        layers = model.set_up_transformer_layers(
            embed_dim, num_heads, depth, dropout, bias, zero_attn
        )
        assert isinstance(
            layers, nn.TransformerEncoder
        ), "`layers` not type(obj) == nn.TransformerEncoder"

        for layer in layers.layers:
            assert hasattr(
                layer.get_submodule('self_attn.out_proj'), "bias"
            ) is bias, f"bias={bias} in argument however bias={not bias} in layer"


    # check if the KH Normal layer initialization for the attn layers works
    # see fn definition for `set_up_transformer_layers` for details and why this is necessary
    assert (
        torch.allclose(
            layer.layers[0].self_attn.out_proj.weight,
            layer.layers[1].self_attn.out_proj.weight,
        )
        is False
    ), "Weights are equal for blocks 0 and 1"
    assert (
        torch.allclose(
            layer.layers[0].self_attn.out_proj.weight,
            layer.layers[2].self_attn.out_proj.weight,
        )
        is False
    ), "Weights are equal for blocks 0 and 2"
    assert (
        torch.allclose(
            layer.layers[1].self_attn.out_proj.weight,
            layer.layers[2].self_attn.out_proj.weight,
        )
        is False
    ), "Weights are equal for blocks 0 and 1"

    random_data = torch.randn(5, 64)
    output = layers(random_data)
    assert output.shape == (5, 64), "Output shape is not (5, 64)"
    assert torch.allclose(output, random_data) is False, "Output is not equal to input"

def test_zinb_proj():
    embed_dim = 64
    n_genes = 500
    eps = 1e-15

    zinb_proj = model.ZINBProj(embed_dim, n_genes, eps)

    assert isinstance(
        zinb_proj, model.ZINBProj
    ), "`zinb_proj` was not type(obj) == model.ZINBProj"

    random_data = torch.randn(5, 64)
    output = zinb_proj(random_data)
    for key, matrix in output.items():
        if key != "gate":
            assert torch.all(
                torch.ge(matrix, 0.0)
            ), f"All values in mu/theta/scale projections should be greater than or equal to 0; eps was set to {eps}"

    zinb = distributions.ZeroInflatedNegativeBinomial(
        mu=output["mu"],
        theta=output["theta"],
        zi_logits=output["gate"],
        scale=output["scale"],
    )
    
    assert isinstance(
        zinb, distributions.ZeroInflatedNegativeBinomial
    ), "`zinb` was not type(obj) == distributions.ZeroInflatedNegativeBinomial"
