# -*- encoding:utf-8 -*-
# author : Guo Yuhe

import torch
import torch.nn as nn

from torch_scatter import scatter_mean

from col_spec_yh.embedding import TabEmbedding, MinimalEmbedding, BertEmbedding
from col_spec_yh.encoder import BertTabEncoder
from col_spec_yh.encode_utils_db import _tbl_idx, _row_idx, _col_idx, _cell_idx, \
    _is_title, _is_header, _is_header_cls, _is_title_cls, _is_tok, _is_cell_cls
# from collections import defaultdict

from demos.test_db_generate_seg_and_mask import check_segs

class TabEncoder(nn.Module):
    def __init__(self, args):
        super(TabEncoder, self).__init__()
        self.embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.vocab))
        self.encoder = BertTabEncoder(args)
        self.option = args.option
        # self.pooling = args.pooling


    def forward(self, src, mask):
        emb = self.embedding(src, mask)  # [8, 64, 768]
        output = self.encoder(emb, mask)  # # [8, 64, 768]
        return output


    def encode(self, src, seg, graph=None): # option: ['table', 'cols', 'rows', 'cells']
        if graph is not None:
            output = self.forward(src, graph)
            seq_len, emb_size = output.shape
        else:
            output = self.forward(src, seg)
            bz, seq_len, emb_size = output.shape

        seg_with_tab_id = False
        if type(seg)==list and int(max(seg)) > 10**6:
            seg_with_tab_id = True

        # output_nl = output[torch.nonzero(_is_nl(seg//100))].view(-1, emb_size)
        # output = output[torch.nonzero(torch.logical_not(_is_nl(seg//100)))].view(-1, emb_size)

        # check_segs(zip([(seg // 100).tolist()], [src.tolist()]), 'row')
        ret = {}
        option = self.option
        # import ipdb;        ipdb.set_trace()
        if seg_with_tab_id:
            if 'tab' in option :# or 'table' in option:
                pooling = option['tab']
                if pooling == 'tab-cls':
                    _ids = torch.nonzero(_is_title_cls(seg // 100))
                    _o = output[_ids]
                    o = _o.view(-1, emb_size)   # (tbl_num, h)
                if pooling == 'avg-title-token':
                    _ids = torch.nonzero(_is_title(seg//100))       # (#instance, seg.dim())
                    _o = output[_ids]   # (#instance, seg.dim(), h)
                    _seg = seg[_ids]

                    _ids_tbl = _tbl_idx(_seg)-1     # (#instance, seg.dim())
                    _o = scatter_mean(_o, index=_ids_tbl, dim=0)    # (#tbl_instance, seg.dim(), h)
                    o = _o.view(-1, emb_size)     # (tbl_num, h)
                ret['tab'] = o

            if 'col' in option:
                pooling = option['col']
                if pooling == 'avg-header-token':
                    _ids = torch.nonzero(_is_header(seg//100))
                    _o = output[_ids]
                    _seg = seg[_ids]

                    _ids_tbl = _tbl_idx(_seg)
                    _ids_col = _col_idx(_seg//100)
                    _ids_tbl_col = _ids_tbl*100 + _ids_col

                    _ids, cnts = _ids_tbl_col.unique(return_counts=True)
                    _ids_new = torch.arange(_ids.shape[0]).to(output.device)
                    _ = []
                    for id_new, cnt in zip(_ids_new.tolist(), cnts.tolist()):
                        _.extend([id_new] * cnt)
                    ids = torch.LongTensor(_).to(output.device)   #(col_tok_num)

                    _o = scatter_mean(_o, ids, dim=0)  # (#col_num, 1, 768)
                    o = _o.view(-1, emb_size)
                if pooling == 'col-cls':
                    _ids = torch.nonzero(_is_header(seg//100))
                    _o = output[_ids]
                    o = _o.view(emb_size, -1)

                # assert emb_cols.shape == (bz, col_num, emb_size)
                ret['cols'] = o

            if 'nl' in option:
                pooling = option['nl']
                if pooling == 'avg-lemma-token':
                    _ids = torch.nonzero(_is_nl_wo_cls(seg//100))
                    _o = output[_ids].view(-1, emb_size)
                    _seg = seg[_ids]

                    _ids = _col_idx(_seg//100)-1
                    o = scatter_mean(_o, _ids, dim=0).view(-1, emb_size)
                ret['lemmas'] = o


        else:
            if 'tab' in option:  # or 'table' in option:
                pooling = option['tab']
                if pooling == 'tab-cls':
                    _ids = torch.nonzero(_is_title_cls(seg))
                    _o = output[_ids.T[0],_ids.T[1],:]
                    o = _o.view(-1, emb_size)  # (tbl_num, h)
                if pooling == 'avg-title-token':
                    _ids = torch.nonzero(_is_title(seg))  # (#instance, seg.dim())
                    _o = output[tuple(_ids.T)] # (#instance, seg.dim(), h)
                    _ids_tbl = _ids.T[0]
                    _o = scatter_mean(_o, index=_ids_tbl, dim=0)  # (#tbl_instance, seg.dim(), h)
                    o = _o.view(-1, emb_size)  # (tbl_num, h)
                ret['tab'] = o

            if 'col' in option:
                import ipdb; ipdb.set_trace()
                pooling = option['col']
                if pooling == 'avg-header-token':
                    _ids = torch.nonzero(_is_header(seg))
                    _o = output[tuple(_ids.T)]
                    _seg = seg[tuple(_ids.T)]

                    _ids_tbl = _ids.T[0]
                    _ids_col = _col_idx(_seg)
                    _ids_tbl_col = _ids_tbl*100 + _ids_col

                    _ids, cnts = _ids_tbl_col.unique(return_counts=True)
                    _ids_new = torch.arange(_ids.shape[0]).to(output.device)
                    _ = []
                    for id_new, cnt in zip(_ids_new.tolist(), cnts.tolist()):
                        _.extend([id_new] * cnt)
                    ids = torch.LongTensor(_).to(output.device)   #(col_tok_num)

                    _o = scatter_mean(_o, ids, dim=0)  # (#col_num, 1, 768)
                    o = _o.view(-1, emb_size)
                if pooling == 'col-cls':
                    _ids = torch.nonzero(_is_header_cls(seg))
                    _o = output[tuple(_ids.T)]
                    o = _o.view(emb_size, -1)
                # assert emb_cols.shape == (bz, col_num, emb_size)
                ret['cols'] = o
            if 'first-column' in option:
                pooling = option['first-column']
                if pooling == 'col-cls':
                    _ids = torch.nonzero(_is_header_cls(seg) * (_col_idx(seg)==1) )
                    o = output[tuple(_ids.T)]               # .view(-1, emb_size)
                if pooling == 'avg-cell-cls':
                    _ids = torch.nonzero(_is_cell_cls(seg) * (_col_idx(seg) == 1))
                    _o = output[tuple(_ids.T)]
                    _ids_tbl = _ids.T[0]
                    o = scatter_mean(_o, _ids_tbl, dim=0)   # .view(-1, emb_size)
                if pooling == 'avg-token':
                    _ids = torch.nonzero(_is_tok(seg) * (_col_idx(seg) == 1))
                    _o = output[tuple(_ids.T)]
                    _ids_tbl = _ids.T[0]
                    o = scatter_mean(_o, _ids_tbl, dim=0)   # .view(-1, emb_size)
                ret['first-column'] = o
            if 'first-row' in option:
                pooling = option['first-row']
                if pooling == 'row-cls':
                    _ids = torch.nonzero(_is_row_cls(seg) * (_row_idx(seg)==1) )
                    o = output[tuple(_ids.T)]               # .view(-1, emb_size)
                if pooling == 'avg-cell-cls':
                    _ids = torch.nonzero(_is_cell_cls(seg) * (_row_idx(seg) == 1))
                    _o = output[tuple(_ids.T)]
                    _ids_tbl = _ids.T[0]
                    o = scatter_mean(_o, _ids_tbl, dim=0)   # .view(-1, emb_size)
                if pooling == 'avg-token':
                    _ids = torch.nonzero(_is_tok(seg) * (_row_idx(seg) == 1))
                    _o = output[tuple(_ids.T)]
                    _ids_tbl = _ids.T[0]
                    o = scatter_mean(_o, _ids_tbl, dim=0)   # .view(-1, emb_size)
                ret['first-row'] = o



        # import ipdb; ipdb.set_trace()
        return ret

        # if option == 'cells':
        #     if self.pooling == 'sep':
        #         pass
        #     if self.pooling == 'avg':
        #         pass

