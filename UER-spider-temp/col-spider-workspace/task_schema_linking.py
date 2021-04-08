from demos.test_db_generate_seg_and_mask import check_segs
from demos.utils import get_args_minimal as get_args
from col_spec_yh.store_utils import decode_sqlite_spider_tbls_in_rows_with_metadata

from col_spec_yh.encode_utils import generate_mask
from col_spec_yh.encode_utils import *
from col_spec_yh.encode_utils_db import f_is_meta
from col_spec_yh.encode_utils_db import generate_seg_with_meta
from col_spec_yh.model import TabEncoder

from demos.utils import load_or_initialize_parameters, build_optimizer
# from uer.utils.optimizers import WarmupLinearSchedule, AdamW

import json
import dgl
import os

from torch import nn
import torch.nn.functional as F

class ct_linker(nn.Module):
    def __init__(self, tabEncoder, link_scorer):
        super(ct_linker, self).__init__()
        self.model = tabEncoder
        self.ct_link_scorer = link_scorer

    def forward(self, nids, seg, graph=None):
        _ = self.model.encode(nids, seg, graph=graph)
        col_n_tabs = torch.vstack((_['tabs'], _['cols']))
        lemmas = _['lemmas']
        lemma_col_tab = _get_all_pair_cat(lemmas, col_n_tabs)  #
        ct_link_scores = self.ct_link_scorer(lemma_col_tab).view(lemmas.size()[0], -1)
        # loss = F.cross_entropy(ct_link_scores, label_ct_link)
        return ct_link_scores



def get_ct_link_scorer(h):
    return nn.Sequential(
        nn.Linear(h * 2, h),
        nn.Tanh(),
        nn.Linear(h, 1)
    )

def _get_total_que_num():
    return 7000
    # ants = json.load(open('./data/spider/tabEnc_train_db.json'))
    # _ = [len(_) for _ in list(ants.values())]
    # return sum(_)


def _get_label(args, ant, tbls):
    # [None, None, {'agg': ['count'], 'func': [], 'id': 1, 'scope': 'main', 'type': 'tbl'}, None, None, None]
    tbl_num = len(tbls)
    col_num = sum([len(tbl['header']) for tbl in tbls.values()])
    col_num_with_sql_all = col_num + 1
    none_id = col_num_with_sql_all + tbl_num
    label = []
    for _ in ant:
        if type(_) == dict and _.get('type'):
            if _['type'] == 'tbl':
                label.append(_['id']+col_num_with_sql_all)
            elif _['type'] == 'col':
                label.append(_['id'])
            else:
                label.append(none_id)
        else :# _ == None:
            label.append(none_id)
    # return F.one_hot(torch.LongTensor(label), num_classes=col_num+tbl_num+2)
    assert len(label) == len(ant)
    label = label * 2 # 改回来！
    return torch.LongTensor(label).to(args.device)

def _get_all_pairs(x, y):
    # x = torch.arange(5)
    # y = torch.arange(10)

    x = x.expand(y.shape[0], x.shape[0]).t()  # expand to (10 ,5), then transpose to (5, 10)
    y = y.expand(x.shape[0], y.shape[0])  # expand to (5, 10)

    x = x.unsqueeze(-1)  # (5, 10, 1)
    y = y.unsqueeze(-1)  # (5, 10, 1)

    return torch.cat([x, y], -1).view(-1,2)

def _get_all_pair_cat(x, y):
    # x.dim() == y.dim() == 2
    xx = x.unsqueeze(1).repeat(1, y.shape[0], 1).view(-1, x.shape[-1])
    yy = y.repeat(x.shape[0], 1, 1).view(-1, y.shape[-1])
    return torch.cat((xx, yy), dim=1)

def _get_db_graph_and_held_out_meta_nids(args, tbls, with_special_col=False):
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

    db_edges = torch.LongTensor(2, 0).to(args.device)
    db_n_idxs = torch.LongTensor(0).to(args.device)
    meta_idxs = torch.LongTensor(0).to(args.device)
    seg_for_db = torch.LongTensor(0).to(args.device)
    max_nid = 0
    if with_special_col:
        SQL_STAR_ID = 7

        db_n_idxs = torch.hstack((db_n_idxs, torch.LongTensor([SQL_STAR_ID]).to(args.device)))
        meta_idxs = torch.hstack((meta_idxs, torch.LongTensor([max_nid]).to(args.device)))
        seg_for_db = torch.hstack((seg_for_db, torch.LongTensor([19800]).to(args.device)))
        max_nid += 1

    for tid, _ in tbls.items():
        rows = _['rows']
        cols = list(zip(*rows))

        tokens, seg = generate_seg_with_meta(args, cols, tbl_id=tid+1,
                                             title=_['title'], header=_['header'],
                                             dgl_backend=True)
        seg = torch.LongTensor(seg).to(args.device)
        tokens = torch.LongTensor(tokens).to(args.device)

        _mask = generate_mask(seg//100, additional_ban=4).squeeze(0).squeeze(0)  # [1, 1, seq_len, seq_len]  --> [seq_len, seq_len]
        _now_tab_edge = _mask.nonzero().T + max_nid  # [[_,_],  ]           [num, 2] -> [2, num]
        _now_meta_idxs = f_is_meta(seg//100).nonzero().T.view(-1) + max_nid

        # update variables for next_loop
        max_nid += len(seg)
        db_edges = torch.hstack((db_edges, _now_tab_edge))
        db_n_idxs = torch.hstack((db_n_idxs, tokens))
        meta_idxs = torch.hstack((meta_idxs, _now_meta_idxs))
        seg_for_db = torch.hstack((seg_for_db, seg))  # dim=2

    if with_special_col:
        SQL_NONE_ID = 6

        max_nid += 1
        db_n_idxs = torch.hstack((db_n_idxs, torch.LongTensor([SQL_NONE_ID]).to(args.device)))
        meta_idxs = torch.hstack((meta_idxs, torch.LongTensor([max_nid]).to(args.device)))
        seg_for_db = torch.hstack((seg_for_db, torch.LongTensor([19898*100+tid+2]).to(args.device)))
    return db_edges, db_n_idxs, meta_idxs, max_nid, seg_for_db


def set_args():
    args = get_args()
    # args.high_level_clses = ['TAB', 'COL', 'CELL', 'NL']
    args.high_level_clses = ['TAB', 'COL', 'NL']
    # args.option = ['tab', 'col']
    # args.pooling = 'avg-token'
    args.option = {'tab': 'avg-title-token',  # or 'tab-cls' || 之后考虑： 'avg-col-cls' 'avg-cont-token'
                   'col': 'avg-header-token',  # or 'col-cls'
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
    args.pretrained_model_path = "./models/bert_model.bin-000"
    args.learning_rate = 2e-5
    args.warmup = 0.1
    args.epoch_num = 20
    # args.pooling = ''
    return args

def que_step_loader(args, tbls, ques):
    # build for tables
    db_edges, db_n_idxs, meta_idxs, tbl_part_max_nid, seg_for_db = \
        _get_db_graph_and_held_out_meta_nids(args, tbls, with_special_col=True)

    # add nl part
    for nl in ques:
        max_nid = tbl_part_max_nid
        nl_n_idxs, seg_for_nl = generate_seg_with_meta(args,cols=[],tbl_id=0, nl=nl+nl,dgl_backend=True)
        # 改回来！
        nl_n_idxs = torch.LongTensor(nl_n_idxs).to(args.device)
        seg_for_nl = torch.LongTensor(seg_for_nl).to(args.device)

        que_nodes = torch.arange(len(nl_n_idxs)).to(args.device)+ max_nid
        que_edges = _get_all_pairs(que_nodes, que_nodes).T
        que_db_edges = _get_all_pairs(meta_idxs, que_nodes).T

        edges = torch.cat([db_edges, que_edges, que_db_edges], dim=1)
        nids = torch.cat([db_n_idxs, nl_n_idxs])
        seg = torch.cat([seg_for_db, seg_for_nl]).to(args.device)
        g = dgl.graph((edges[1], edges[0]))
        print(g)
        yield g, nids, seg


def train_step_db(args, optimizer, scheduler, model, tbls, ques, label_ct_link):
    for (g, nids, seg), label_ct_link in zip(que_step_loader(args, tbls, ques), label_ct_link):
        model.zero_grad()
        ct_link_scores = model(nids, seg, graph=g)
        loss = F.cross_entropy(ct_link_scores, label_ct_link)
        loss.backward()
        optimizer.step()
        scheduler.step()
        yield loss
        # print(loss.item())
        # print('loss: {}'.format(loss.item()))

def train_loader(args, epoch_num):
    db_info_path = './data/spider/processed_db.json'
    ants = json.load(open('./data/spider/tabEnc_train_db.json'))
    # db_base_path = './data/spider/slim_99/'
    db_base_path = './data/spider/slim_test_10/'

    train_db_id_list = list(ants.keys())
    import random
    for epoch_id in range(epoch_num):
        random.shuffle(train_db_id_list)
        for db_id in train_db_id_list:
            db_path = os.path.join(db_base_path, db_id, db_id + '.sqlite')
            # import ipdb; ipdb.set_trace()
            tbls = decode_sqlite_spider_tbls_in_rows_with_metadata(db_path, db_info_path)
            ants_db = ants[db_id]
            ques = [_['lemma'] for _ in ants_db]
            labels = [_get_label(args, _['ant'], tbls) for _ in ants_db]
            print('db_id: {}; que_num: {}'.format(db_id, len(ques)))
            yield epoch_id, tbls, ques, labels

def evaluate(model, ct_link_scorer):
    return




def main():
    args = set_args()
    args.device='cuda:1'
    ct_link_scorer = get_ct_link_scorer(args.hidden_size)
    tab_encoder = TabEncoder(args)

    model = ct_linker(tab_encoder, ct_link_scorer)
    model.to(args.device)

    args.train_steps = args.epoch_num*_get_total_que_num()
    load_or_initialize_parameters(args, model.model)

    # train
    optimizer, scheduler = build_optimizer(args, model)
    now_epoch = 0
    for db_num, (epoch_id, tbls, ques, labels) in enumerate(train_loader(args, args.epoch_num)):
        if epoch_id > now_epoch:
            now_epoch += 1
            # TODO: logging, evaluate()
        for loss in train_step_db(args, optimizer, scheduler, model, tbls, ques, labels):
            print('Epoch: {}, db_num: {}, loss:{}'.format(epoch_id, db_num,loss))
            print('Epoch: {}, db_num: {}, loss:{}'.format(epoch_id, db_num, loss), file=open('recc','a'))
            break  # 改回来

    return

if __name__=='__main__':
    # train_loader()
    main()
