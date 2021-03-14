import os
import sys
from sklearn.utils import Bunch

import logging  # 引入logging模块
import time
import torch

from uer.utils.vocab import Vocab
from uer.utils.tokenizer import BertTokenizer

from col_spec_yh.store_utils import get_labels_map_from_aida_file_2

def get_logger(logger_name="detail", dir_name='logs_default', file_name='log_rec_all'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Log等级总开关

    if not os.path.exists(dir_name) or os.path.isfile(dir_name):
        os.makedirs(dir_name)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    detail_log_name = os.path.join(dir_name, rq) + '.log'

    # add a file handler
    fh = logging.FileHandler(detail_log_name, mode='w')
    fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
    formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # add a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # DEBUG < INFO （more strict）
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # another file handler
    fh_2 = logging.FileHandler(file_name, mode='a')
    fh_2.setLevel(logging.WARNING)  # 输出到file的log等级的开关
    formatter = logging.Formatter("%(asctime)s - %(filename)s [#l:%(lineno)d] - %(message)s")
    fh_2.setFormatter(formatter)
    logger.addHandler(fh_2)
    return logger


def get_args_aida_task():
    args = Bunch()
    args.seq_len = 64
    args.row_wise_fill = True
    args.mask_mode = 'cross-wise'
    args.additional_ban = 2
    args.table_object = 'first-column'
    args.pooling = 'avg-token'

    args.pretrained_model_path = "./models/bert_model.bin-000"
    args.vocab_path = 'models/google_uncased_en_vocab.txt'
    args.vocab = Vocab()
    args.vocab.load(args.vocab_path)
    args.emb_size = 768
    args.embedding = 'tab'  # before: bert
    args.encoder = 'bertTab'
    args.subword_type = 'none'
    args.tokenizer = 'bert'
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    args.feedforward_size = 3072
    args.hidden_size = 768
    args.heads_num = 12
    args.layers_num = 12
    args.learning_rate = 2e-5
    args.batch_size = 4
    args.dropout = 0.1

    # args.target = 'bert'
    return args


def get_args_minimal():
    args = Bunch()
    args.vocab_path = 'models/google_uncased_en_vocab.txt'
    args.vocab = Vocab()
    args.vocab.load(args.vocab_path)
    args.tokenizer = 'bert'
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # 'pooling': 'avg-cell-seg',  # avg-token, 'avg-cell-seg'
    args.mask_mode = 'cross-wise'
    # 'shuffle_rows': True,
    args.additional_ban = 4  # ?
    # 'row_wise_fill': False,
    args.has_high_level_cls = True
    return args

def set_args_2():
    # options for model
    args = Bunch()
    args.mask_mode = 'cross-wise'  # in ['row_wise', 'col_wise', 'cross_wise', 'cross_and_hier_wise']
    args.additional_ban = 0
    # args.pooling = 'avg-token'
    args.pooling = 'avg-cell-seg'
    args.table_object = 'first-column'
    args.noise_num = 2
    args.seq_len = 100
    args.row_wise_fill = True

    args.pretrained_model_path = "./models/bert_model.bin-000"
    args.vocab_path = 'models/google_uncased_en_vocab.txt'
    args.vocab = Vocab()
    args.vocab.load(args.vocab_path)
    args.emb_size = 768
    args.embedding = 'tab'  # before: bert
    args.encoder = 'bertTab'
    args.subword_type = 'none'
    args.tokenizer = 'bert'

    args.feedforward_size = 3072
    args.hidden_size = 768
    args.heads_num = 12
    args.layers_num = 12
    args.learning_rate = 2e-5
    args.warmup = 0.1
    args.batch_size = 32
    args.dropout = 0.1
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.train_path = './data/aida/IO/train_samples'
    args.t2d_path = './data/aida/IO/test_samples_t2d'
    args.limaye_path = './data/aida/IO/test_samples_limaye'
    args.wiki_path = './data/aida/IO/test_samples_wikipedia'


    # other options
    args.report_steps = 100
    args.labels_map = get_labels_map_from_aida_file_2(args.train_path)
    args.labels_num = len(args.labels_map)
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)
    return args


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        print("[YH INFO] : Loading pretrained parameters from {}.".format(args.pretrained_model_path))
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, tgt_batch, seg_batch