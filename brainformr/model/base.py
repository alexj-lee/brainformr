from typing import Optional

import torch
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
              put_device: Optional[str] = 'cuda',
              eps: float = 1e-15,
              n_genes: Optional[int] = 500,
              xformer_dropout: Optional[float] = 0.0,
              bias: Optional[bool] = True,
              zero_attn: Optional[bool] = True,
                ):
        """celltransformer secondarry docstring

        Parameters
        ----------
        encoder_embedding_dim : int
            _description_
        encoder_num_heads : int
            _description_
        encoder_depth : int
            _description_
        decoder_embedding_dim : int
            _description_
        decoder_num_heads : int
            _description_
        decoder_depth : int
            _description_
        attn_pool_heads : Optional[int], optional
            _description_, by default 8
        cell_cardinality : Optional[int], optional
            _description_, by default 1024
        put_device : Optional[str], optional
            _description_, by default 'cuda'
        eps : float, optional
            _description_, by default 1e-15
        n_genes : Optional[int], optional
            _description_, by default 500
        xformer_dropout : Optional[float], optional
            _description_, by default 0.0
        bias : Optional[bool], optional
            _description_, by default True
        zero_attn : Optional[bool], optional
            _description_, by default True
        """
        super(CellTransformer, self).__init__()
    
        self.eps = eps

        _feature_dim = encoder_embedding_dim // 2
        self.expression_projection = get_projection_layers(n_genes, _feature_dim)
        self.encoder_cell_embed = nn.Embedding(cell_cardinality, _feature_dim)

        self.decoder_cell_embed = nn.Embedding(cell_cardinality, decoder_embedding_dim)
        self.attn_pool = AttnPool(encoder_embedding_dim, attn_pool_heads)

        self.cls_token = nn.Parameter(torch.zeros(1, encoder_embedding_dim))

        self.encoder = set_up_transformer_layers(encoder_embedding_dim, encoder_num_heads, encoder_depth, xformer_dropout, bias, zero_attn)
        self.decoder = set_up_transformer_layers(decoder_embedding_dim, decoder_num_heads, decoder_depth, xformer_dropout, bias, zero_attn)

        self.zinb_proj = ZINBProj(embed_dim=decoder_embedding_dim, n_genes=n_genes, eps=self.eps)   

        self.put_device = put_device

    def forward(self, data_dict: dict):
        bs = data_dict['bs']

        cells = data_dict["observed_cell_type"].to(self.put_device, non_blocking=False)
        encoder_mask = data_dict["encoder_mask"].to(self.put_device, non_blocking=False)
        num_hidden = len(data_dict['masked_cell_type']) # will in practice be == bs but in case later want to train on multiple cells
        hidden_expression = data_dict['masked_expression'].to(self.put_device, non_blocking=False)
        hidden_cells = data_dict['masked_cell_type'].to(self.put_device, non_blocking=False)
        pooling_mask = data_dict['pooling_mask'].to(self.put_device, non_blocking=False)
        decoder_mask = data_dict['decoder_mask'].to(self.put_device, non_blocking=False)

        expression = data_dict["observed_expression"].to(
            self.put_device, non_blocking=False
        )

        cls_tokens = self.cls_token.repeat_interleave(bs, dim=0)

        if len(expression) > 0:
            cells_embed = self.encoder_cell_embed(cells)
            # expression_embed = self.expression_projection(expression[:, :500])
            expression_embed = self.expression_projection(expression)
            cells_embed = torch.cat((cells_embed, expression_embed), dim=1)
            cells_embed = torch.cat((cells_embed, cls_tokens), dim=0)
        else:
            cells_embed = cls_tokens

        cells_embed = self.encoder(cells_embed, mask=encoder_mask)

        decoding_queries = self.decoder_cell_embed(hidden_cells)

        cells_embed = self.attn_pool(cells_embed, pooling_mask, bs)

        cells_embed = torch.cat((cells_embed, decoding_queries), dim=0
                                )        

        cells_embed = self.decoder(cells_embed, mask=decoder_mask)
        ref_cell_embed = cells_embed[-num_hidden:]
    
        zinb_params = self.zinb_proj(ref_cell_embed)

        cls_toks = cells_embed[:-num_hidden]

        return dict(
            zinb_params=zinb_params,
            neighborhood_repr=cls_toks,
            hidden_expression=hidden_expression
        )








    
    