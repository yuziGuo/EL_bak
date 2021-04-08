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

# 可以改成其它 train_loader
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
            yield epoch_id, tbls



def set_args():
    args = get_args()
    args.high_level_clses = ['TAB', 'COL', 'CELL', 'ROW','NL']
    # args.high_level_clses = ['TAB','COL','NL']
    # args.option = ['tab', 'col']
    # args.pooling = 'avg-token'
    args.option = {
        'tab': 'avg-title-token'  # or 'tab-cls' || 之后考虑： 'avg-col-cls' 'avg-cont-token'
        # ,'col': 'avg-header-token'  # or 'col-cls'
        # ,'col': 'col-cls'

        # , 'first-column': 'col-cls'
        # , 'first-column': 'avg-token'
        , 'first-column': 'avg-cell-cls'
        # , 'nl': 'avg-lemma-token'
        }
    args.dgl_backend = False
    args.seq_len = 100
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


def main():
    # model
    args = set_args()
    args.device='cuda:1'
    model = TabEncoder(args)
    model.to(args.device)

    for epoch_id, tbls in train_loader(args, epoch_num=1):
        # 本 demo 只用 tbls
        # generate_seg_with_meta

        tokens, segs = [], []
        for tid, _ in tbls.items():
            rows = _['rows']
            cols = list(zip(*rows))
            toks, seg = generate_seg_with_meta(args, cols  # args.dgl_back_end
                                                 # , tbl_id=tid + 1,
                                                 , title=_['title'], header=_['header']
                                                , row_wise_fill = False
                                               )
            check_segs(zip([seg], [toks]), 'col')
            tokens.append(toks)
            segs.append(seg)

        segs = torch.LongTensor(segs).to(args.device)
        tokens = torch.LongTensor(tokens).to(args.device) # [t_num, args.seq_len]

        # import ipdb; ipdb.set_trace()
        _ = model.encode(tokens, segs)



if __name__ == '__main__':
    main()
