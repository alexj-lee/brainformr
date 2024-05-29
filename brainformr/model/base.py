from typing import Optional

from torch import nn

from .blocks import AttnPool, ZINBProj
from .factory import get_projection_layers, set_up_transformer_layers


class CellTransformer(nn.Module):
    def __init__(self,
              encoder_embedding_dim: int,
              encoder_num_heads: int,
              encoder_depth: int,
              decoder_embedding_dim: int,
              decoder_num_heads: int,
              decoder_depth: int,
              attn_pool_heads: Optional[int] = 8,
              cell_cardinality: Optional[int] = 1024,
              device: Optional[str] = 'cuda',
              eps: float = 1e-15,
              n_genes: Optional[int] = 500,
              xformer_dropout: Optional[float] = 0.0,
              bias: Optional[bool] = True,
              zero_attn: Optional[bool] = True,
              ):
    
        self.eps = eps

        _feature_dim = encoder_embedding_dim // 2
        self.expression_projection = get_projection_layers(n_genes, _feature_dim)
        self.encoder_cell_embed = nn.Embedding(cell_cardinality, _feature_dim)

        self.decoder_cell_embed = nn.Embedding(cell_cardinality, decoder_embedding_dim)
        self.attn_pool = AttnPool(encoder_embedding_dim, attn_pool_heads)

        self.encoder = set_up_transformer_layers(encoder_embedding_dim, encoder_num_heads, encoder_depth, xformer_dropout, bias, zero_attn)
        self.decoder = set_up_transformer_layers(decoder_embedding_dim, decoder_num_heads, decoder_depth, xformer_dropout, bias, zero_attn)

        self.zinb_proj = ZINBProj(embed_dim=decoder_embedding_dim, n_genes=n_genes, eps=self.eps)   

    def forward(self, x):
        pass





    
    