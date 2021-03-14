# -*- encoding:utf-8 -*-
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.transformer import TransformerLayer, RelationAwareTransformerLayer
import torch
import time

from col_spec_yh.encode_utils import generate_mask_crosswise

class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.mask_mode = args.mask_mode
        if args.mask_mode=='crosswise_rel':
            hidden_size = args.emb_size // args.heads_num
            relations = {'masked':0, 'col':1, 'row':2, 'in_cell':3}
            self.relationEmbedding_K = nn.Embedding(len(relations), hidden_size)
            self.relationEmbedding_V = nn.Embedding(len(relations), hidden_size)
            self.transformer = nn.ModuleList([
                RelationAwareTransformerLayer(args) for _ in range(self.layers_num)
            ])
        else:
            self.transformer = nn.ModuleList([
                TransformerLayer(args) for _ in range(self.layers_num)
            ])

    def get_mask_origin(self, seg, seq_length):
        """
       Args:
           seg: [batch_size x seq_length]

       Returns:
           mask: [batch_size x 1 x seq_length x seq_length]
       """
        mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)
        return mask.float()

    # def get_mask_arch(self, seg, seq_length):
    #     """
    #    Args:
    #        seg: [batch_size x seq_length]
    #
    #    Returns:
    #        mask: [batch_size x 1 x seq_length x seq_length]
    #    """
    #     half_len = seq_length // 2
    #     mask_a = (seg == 1).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
    #     mask_a = mask_a[:,:,:,:half_len]
    #     mask_b = (seg == 2).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
    #     mask_b = mask_b[:, :, :, half_len:]
    #     mask_zeros = torch.zeros(mask_a.shape, device=mask_a.device)
    #     mask_a = torch.cat([mask_a,mask_zeros],dim=-1)
    #     mask_b = torch.cat([mask_zeros,mask_b], dim=-1)  # [16, 1, 128, 128]
    #     mask_a = mask_a[:,:,:half_len,:]
    #     mask_b = mask_b[:,:,half_len:,:]
    #     # import ipdb;ipdb.set_trace()
    #     mask_new = torch.cat([mask_a, mask_b], dim=-2) # [16,1,128,128]
    #     return mask_new

    def get_mask_arch(self, seg):
        mask = torch.stack([
            (
                    (s.unsqueeze(1) == s).float() * \
                    (s.unsqueeze(1) > 0).float() * (s > 0).float()
            ) > 0
            for s in seg
        ]).unsqueeze(1).float()
        return mask

    def get_mask_set_transformer(self, seg, seq_length):
        # 一级公民和在同一一级公民下的二级公民可以相互看到(数字相同或相补)
        def get_mask_npsee(seg):
            mask = torch.stack(
                [
                    torch.stack([abs(_) == abs(s) for _ in s])
                    for s in seg
                ]
            ).unsqueeze(1)
            return mask

        # 一级公民之间可以相互看到(数字相同或相补)
        def get_mask_psee(seg):
            mask = torch.stack(
                [
                    torch.stack(
                        [
                            torch.stack([s[idx] > 0 and ele > 0 for ele in s]) for idx in range(len(seg[0]))
                        ])
                    for s in seg
                ]
            ).unsqueeze(1)
            return mask

        mask_1 = get_mask_npsee(seg)
        mask_2 = get_mask_psee(seg)

        return (mask_1 + mask_2) > 0


    def get_mask_set_transformer_opt(self, seg):
        return torch.stack([
            (
                    (s.unsqueeze(1) == s).float() + \
                    (s.unsqueeze(1) + s == 0).float() + \
                    (s.unsqueeze(1) > 0).float() * (s > 0).float()
            ) > 0
            for s in seg
        ]).unsqueeze(1)

    def get_mask_crosswise(self, seg):
        # elements in generated mask are: 0 1 2 3..
        # seg = torch.Tensor(seg)  # [b,l]
        b, l = seg.shape
        seg_2 = seg.view(b, l, 1)
        seg = seg.unsqueeze(-2)  # [b, 1, l]
        row_wise_see = (seg % 100 == seg_2 % 100).unsqueeze(1).float()  # mask: [batch_size x 1 x seq_length x seq_length]
        col_wise_see = (seg // 100 == seg_2 // 100).unsqueeze(1).float()*2
        mask = row_wise_see + col_wise_see
        return mask

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        # get new_mask

        # no arch!
        # import ipdb
        # ipdb.set_trace()
        # print('Seg shape: ', seg.size())
        # % now_time = time.time()
        # % print('Start calc mask: ', time.strftime("[%y%m%d-%H%M%S]", time.localtime()))
        # mask = self.get_mask_set_transformer(seg, seq_length)
        # mask = self.get_mask_origin(seg, seq_length)
        # mask = self.get_mask_set_transformer_opt(seg)
        # % print('Elapse time for calc mask (s) : ', time.time()-now_time)
        if self.mask_mode == 'arch':
            mask = self.get_mask_arch(seg)
            mask = (1.0 - mask) * -10000.0
            mask_2 = self.get_mask_origin(seg, seq_length)
            mask_2 = (1.0 - mask_2) * -10000.0
            hidden = emb
            # self.layers_num = 4
            layers = list(range(self.layers_num))
            for i in layers[:-1]:
                hidden = self.transformer[i](hidden, mask)
            hidden = self.transformer[-1](hidden, mask_2)
        if self.mask_mode == 'arch_2mix_top':
            mask = self.get_mask_arch(seg)
            mask = (1.0 - mask) * -10000.0
            mask_2 = self.get_mask_origin(seg, seq_length)
            mask_2 = (1.0 - mask_2) * -10000.0
            hidden = emb
            # self.layers_num = 4
            layers = list(range(self.layers_num))
            for i in layers[:-2]:
                hidden = self.transformer[i](hidden, mask)
            for i in layers[-2:]:
                hidden = self.transformer[i](hidden, mask_2)
            # hidden = self.transformer[-1](hidden, mask_2)
        if self.mask_mode == 'arch_reversed':
            mask = self.get_mask_arch(seg)
            mask = (1.0 - mask) * -10000.0
            mask_2 = self.get_mask_origin(seg, seq_length)
            mask_2 = (1.0 - mask_2) * -10000.0
            hidden = emb
            # self.layers_num = 4
            layers = list(range(self.layers_num))
            for i in layers[:3]:
                hidden = self.transformer[i](hidden, mask_2)
            for i in layers[3:]:
                hidden = self.transformer[i](hidden, mask)
        if self.mask_mode == 'origin':
            mask = self.get_mask_origin(seg, seq_length)
            mask = (1.0 - mask) * -10000.0
            hidden = emb
            # self.layers_num = 4
            layers = list(range(self.layers_num))
            for i in layers:
                hidden = self.transformer[i](hidden, mask)
        if self.mask_mode == 'crosswise':
            mask = generate_mask_crosswise(seg)
            mask = (mask > 0).float()
            mask = (1.0 - mask) * -10000.0
            hidden = emb
            layers = list(range(self.layers_num))
            for i in layers:
                hidden = self.transformer[i](hidden, mask)  # in mask: 0,1,2,3

        if self.mask_mode == 'crosswise_rel':
            mask = self.get_mask_crosswise(seg)
            # mask = (1.0 - mask) * -10000.0
            hidden = emb
            # self.layers_num = 4
            layers = list(range(self.layers_num))
            for i in layers:
                hidden = self.transformer[i](hidden, mask, self.relationEmbedding_K, self.relationEmbedding_V)  # in mask: 0,1,2,3

        return hidden
