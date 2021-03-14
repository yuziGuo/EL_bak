from demos.test_db_generate_seg_and_mask import check_segs
from demos.utils import get_args_minimal as get_args
from col_spec_yh.store_utils import decode_sqlite_spider_tbls_in_rows_with_metadata

from col_spec_yh.encode_utils import generate_mask
from col_spec_yh.encode_utils import *
from col_spec_yh.encode_utils_db import f_is_meta
from col_spec_yh.encode_utils_db import generate_seg_with_meta
from col_spec_yh.model import TabEncoder

import json
import dgl


def _get_all_pairs(x, y):
    # x = torch.arange(5)
    # y = torch.arange(10)

    x = x.expand(y.shape[0], x.shape[0]).t()  # expand to (10 ,5), then transpose to (5, 10)
    y = y.expand(x.shape[0], y.shape[0])  # expand to (5, 10)

    x = x.unsqueeze(-1)  # (5, 10, 1)
    y = y.unsqueeze(-1)  # (5, 10, 1)

    return torch.cat([x, y], -1).view(-1,2)



def _get_db_graph_and_held_out_meta_nids(args, tbls):
    '''
    :param tbls:
    :return:
        db_edges :
            - Part I:   edges of all separate tables by rule cross-wise-see(ban 4)
            - Part II:  edges that link all separate tables together by meta toks
            - Shape:    torch.LongTensor, shape: (2, #e)
        db_nids :       torch.LongTensor, shape(#n)
        meta_idxs :     torch.LongTensor, shape: (#m)
        max_nid : int
    '''

    max_nid = 0
    db_edges = torch.LongTensor(2, 0)
    db_n_idxs = torch.LongTensor(0)
    meta_idxs = torch.LongTensor(0)
    seg_for_db = torch.LongTensor(0)
    for tid, _ in tbls.items():
        rows = _['rows']
        cols = list(zip(*rows))

        tokens, seg = generate_seg_with_meta(args, cols, tbl_id=tid+1,
                                             title=_['title'], header=_['header'],
                                             dgl_backend=True)
        seg = torch.LongTensor(seg)
        tokens = torch.LongTensor(tokens)

        _mask = generate_mask(seg//100, additional_ban=4).squeeze(0).squeeze(0)  # [1, 1, seq_len, seq_len]  --> [seq_len, seq_len]
        _now_tab_edge = _mask.nonzero().T + max_nid  # [[_,_],  ]           [num, 2] -> [2, num]
        _now_meta_idxs = f_is_meta(seg//100).nonzero().T.view(-1) + max_nid

        # update variables for next_loop
        max_nid += len(seg)
        db_edges = torch.hstack((db_edges, _now_tab_edge))
        db_n_idxs = torch.hstack((db_n_idxs, tokens))
        meta_idxs = torch.hstack((meta_idxs, _now_meta_idxs))
        seg_for_db = torch.hstack((seg_for_db, seg))  # dim=2
    return db_edges, db_n_idxs, meta_idxs, max_nid, seg_for_db


def get_graph_nid_seg(args, db_id, tbls):
    db_edges, db_n_idxs, meta_idxs, max_nid, seg_for_db = _get_db_graph_and_held_out_meta_nids(args, tbls)
    train_db_ants = json.load(open('./data/spider/tabEnc_train_db.json'))[db_id]
    for _ in train_db_ants:
        nl = _['lemma']
        nl_n_idxs, seg_for_nl = generate_seg_with_meta(args,cols=[],tbl_id=0, nl=nl,dgl_backend=True)
        nl_n_idxs = torch.LongTensor(nl_n_idxs)
        seg_for_nl = torch.LongTensor(seg_for_nl)

        # ids
        # meta_idxs = torch.cat([meta_idxs, torch.arange(len(nl))+ max_nid])

        que_nodes = torch.arange(len(nl_n_idxs))+ max_nid
        que_edges = _get_all_pairs(que_nodes, que_nodes).T
        que_db_edges = _get_all_pairs(meta_idxs, que_nodes).T

        edges = torch.cat([db_edges, que_edges, que_db_edges], dim=1)
        nids = torch.cat([db_n_idxs, nl_n_idxs])
        seg = torch.cat([seg_for_db, seg_for_nl])
        g = dgl.graph((edges[1], edges[0]))

        return g, nids, seg



def test_1():
    args = get_args()
    args.high_level_clses = ['TAB', 'COL', 'CELL', 'NL']
    # args.option = ['tab', 'col']
    # args.pooling = 'avg-token'
    args.option = {'tab': 'avg-title-token', # or 'tab-cls' || 之后考虑： 'avg-col-cls' 'avg-cont-token'
                   'col':  'avg-header-token', # or 'col-cls'
                    'nl': 'avg-lemma-token'
                   }
    args.dgl_backend = True
    args.embedding = 'Minimal'
    args.encoder = 'Tab'
    args.dropout = 0.1
    args.emb_size = 768
    args.hidden_size = 768
    args.heads_num = 6
    args.feedforward_size = 3072
    args.layers_num = 12
    # args.pooling = ''

    db_id = 'farm'
    db_path = './data/spider/slim_99/farm/farm.sqlite'
    _ = './data/spider/processed_db.json'
    tbls = decode_sqlite_spider_tbls_in_rows_with_metadata(db_path, _)

    g, nids, seg = get_graph_nid_seg(args, db_id, tbls)
    model = TabEncoder(args)
    _ = model.encode(nids, seg, graph=g)

    tabs = _['tabs']
    cols = _['cols']
    lemmas = _['lemmas']
    import ipdb; ipdb.set_trace()

    return

if __name__=='__main__':
    test_1()