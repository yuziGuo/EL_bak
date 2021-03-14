# -*- encoding:utf-8 -*-
# Author: Guo Yuhe

import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.transformer import TransformerLayer
# from uer.layers.transformer import RelationAwareTransformerLayer

import torch
import time

from col_spec_yh.encode_utils import generate_mask

class BertTabEncoder(nn.Module):
    """
    TODO
    """
    def __init__(self, args):
        super(BertTabEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.mask_mode = args.mask_mode
        self.additional_ban = args.additional_ban
        self.transformer = nn.ModuleList([
            TransformerLayer(args) for _ in range(self.layers_num)
        ])
        # if args.mask_mode=='crosswise_rel':
        #     hidden_size = args.emb_size // args.heads_num
        #     relations = {'masked':0, 'col':1, 'row':2, 'in_cell':3}
        #     self.relationEmbedding_K = nn.Embedding(len(relations), hidden_size)
        #     self.relationEmbedding_V = nn.Embedding(len(relations), hidden_size)
        #     self.transformer = nn.ModuleList([
        #         RelationAwareTransformerLayer(args) for _ in range(self.layers_num)
        #     ])
        # else:
        #     self.transformer = nn.ModuleList([
        #         TransformerLayer(args) for _ in range(self.layers_num)
        #     ])

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]

        assert self.mask_mode in ['row-wise', 'col-wise', 'cross-wise', 'cross-and-hier-wise']
        mask = generate_mask(seg, self.mask_mode, self.additional_ban)
        mask = (mask > 0).float()
        mask = (1.0 - mask) * -10000.0
        hidden = emb
        layers = list(range(self.layers_num))
        for i in layers:
            hidden = self.transformer[i](hidden, mask)

        #  TODO
        # if self.mask_mode == 'crosswise_rel':
        #     mask = self.get_mask_crosswise(seg)
        #     # mask = (1.0 - mask) * -10000.0
        #     hidden = emb
        #     # self.layers_num = 4
        #     layers = list(range(self.layers_num))
        #     for i in layers:
        #         hidden = self.transformer[i](hidden, mask, self.relationEmbedding_K, self.relationEmbedding_V)  # in mask: 0,1,2,3

        return hidden
