from col_spec_yh.encode_utils import generate_seg
from col_spec_yh.encode_utils import generate_mask
from col_spec_yh.model import TabEncoder

from sklearn.utils import Bunch
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, confusion_matrix
from uer.utils.vocab import Vocab
from uer.utils.tokenizer import BertTokenizer
from uer.utils.optimizers import WarmupLinearSchedule, AdamW

from demos.samples.sample_mini_tables import table_a, table_b
# from demos.utils import get_args
from demos.utils import load_or_initialize_parameters
from demos.utils import batch_loader
from demos.utils import get_logger

import logging
import random
import os
from collections import defaultdict
import time
import json

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from col_spec_yh.store_utils import decode_and_verify_aida_file_2, get_labels_map_from_aida_file_2
from col_cls_workspace.data import microTableDataset, fn_wrapper


def set_args(predefined_dict_groups):
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
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    args.feedforward_size = 3072
    args.hidden_size = 768
    args.heads_num = 12
    args.layers_num = 12
    args.learning_rate = 2e-5
    # args.learning_rate = 1e-4
    args.warmup = 0.1
    args.batch_size = 32
    args.dropout = 0.1
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # other options
    args.shuffle_rows = True
    args.report_steps = 100
    for predefined_dict_group in predefined_dict_groups.values():
        for k, v in predefined_dict_group.items():
            args[k] = v
    args.labels_map = get_labels_map_from_aida_file_2(args.train_path)
    args.labels_num = len(args.labels_map)

    # logger and tensorboard writer
    if args.tx_logger_dir_name:
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        args.summary_writer = SummaryWriter(logdir=os.path.join(args.tx_logger_dir_name, '-'.join([args.exp_name,rq])))
    else:
        args.summary_writer = None
    if args.logger_dir_name is not None:
        args.logger_name = 'detail'
        args.logger = get_logger(logger_name=args.logger_name, dir_name=args.logger_dir_name, file_name=args.logger_file_name)
    else:
        args.logger = None
    return args


class col_classifier(nn.Module):
    def __init__(self, args):
        super(col_classifier, self).__init__()
        self.labels_num = args.labels_num
        self.encoder = TabEncoder(args)
        self.table_object = args.table_object
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.criterion = nn.NLLLoss()

    def forward(self, src, seg, tgt=None):
        # output = self.encoder.encode(src, seg, option='first-column')
        # import ipdb; ipdb.set_trace()
        output = self.encoder.encode(src, seg, option=self.table_object)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            loss = self.criterion(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


def build_optimizer(args, model):
    from uer.utils.optimizers import WarmupLinearSchedule, AdamW
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.train_steps*args.warmup, t_total=args.train_steps)
    return optimizer, scheduler


def train_batch(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch):
    model.zero_grad()
    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    loss, _ = model(src_batch, seg_batch, tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss


def eval_batch(args, model, src_batch, seg_batch):
    # model.zero_grad()
    src_batch = src_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    _, logits = model(src_batch, seg_batch)
    return logits


def evaluate(args, model, epoch_id=None):
    def reduce_to_tab_level(logits_all, ds):
        raw_tab_to_sample_id = defaultdict(list)
        for sample_id, raw_tab_id in enumerate([_[0] for _ in ds.samples]):
            raw_tab_to_sample_id[raw_tab_id].append(sample_id)  # rawtab_to_sample_id : {'<raw_tab_id>': [0,1,2,...]}
        raw_tab_ids = list(ds.tb_to_cls_name.keys())
        tab_level_ground_truth = torch.LongTensor([args.labels_map[ds.tb_to_cls_name[tid]] for tid in raw_tab_ids])
        tab_level_logits = torch.stack(
            [
                torch.mean(logits_all[raw_tab_to_sample_id[raw_tab_id]], dim=0)
                for raw_tab_id in raw_tab_ids
            ]
        )
        tab_level_preds = torch.argmax(tab_level_logits, dim=1)
        return tab_level_ground_truth, tab_level_preds

    def get_result_for_one_test_set(args, ds_path, epoch_id):
        ds = microTableDataset(ds_path, train=False, shuffle_rows=args.shuffle_rows)
        _loader = DataLoader(ds, args.batch_size, collate_fn=fn_wrapper(args))
        # ds: list < tokens, label, seg, raw_tab_id >
        with torch.no_grad():
            logits_all = torch.cat(
                [
                    eval_batch(args, model, src_batch, seg_batch)
                    for src_batch, _, seg_batch, _ in _loader
                ]
            )
        tab_level_ground_truth, tab_level_preds = reduce_to_tab_level(logits_all, _loader.dataset)
        acc_score = accuracy_score(tab_level_ground_truth.data.cpu().numpy(),
                                   tab_level_preds.data.cpu().numpy())
        ds_name = os.path.basename(ds_path).split('_')[-1]
        if args.logger:
            args.logger.warning("Epoch_id: {}\t DataSet: {}\tAcc: {}".format(
                epoch_id, ds_name, acc_score
            ))
        return ds_name, acc_score

    results = defaultdict(float)
    for ds_path in [args.t2d_path,args.wiki_path, args.limaye_path]:
        ds_name, acc_score = get_result_for_one_test_set(args, ds_path, epoch_id)
        results[ds_name] = acc_score

    if args.summary_writer:
        args.summary_writer.add_scalars('data/acc', results, epoch_id)


def train_and_eval(args, model):
    ds = microTableDataset(args.train_path, train=True)
    _loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=fn_wrapper(args))
    args.train_steps = int(len(ds) * args.epochs_num / args.batch_size) + 1
    optimizer, scheduler = build_optimizer(args, model)

    for epoch in range(1, args.epochs_num+1):
        total_loss = 0.
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(_loader):
            loss = train_batch(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:

                if args.logger:
                    args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                  total_loss / args.report_steps))
                else:
                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                  total_loss / args.report_steps))

                if args.summary_writer:
                    args.summary_writer.add_scalar('data/loss', total_loss / args.report_steps, epoch)
                total_loss = 0.

        model.eval()
        if epoch >= 1:
            evaluate(args, model, epoch_id=epoch)


def post_experiment(args, model):
    # record logger
    if args.logger:
        logging.getLogger(args.logger_name).handlers = []

    # write things to writer
    # parameters
    # embeddings

    # close summary writer
    if args.summary_writer:
        path = "./acc_and_loss.json"
        # args.summary_writer.export_scalars_to_json(path)
        with open(path, 'a') as f:
            json.dump(args.summary_writer.scalar_dict, f)
            f.write('\n')
        args.summary_writer.close()
        

def experiment(repeat_time, predefined_dict_groups=None):
    for _ in range(repeat_time):
        args = set_args(predefined_dict_groups=predefined_dict_groups)
        # import ipdb; ipdb.set_trace()
        if args.logger:
            args.logger.warning('[For this run] Predefined_dict_groups: {}'.format(predefined_dict_groups))
            args.logger.info('Args: {}'.format(args))
        if args.summary_writer:
            args.summary_writer.add_text('Text', 'Args: {}'.format(args), _)

        model = col_classifier(args)
        load_or_initialize_parameters(args, model.encoder)
        model = model.to(args.device)
        if torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        if args.logger:
            args.logger.info('Model sent to device: {}/{}'.format(model.state_dict()['output_layer_2.bias'].device, args.device))
        train_and_eval(args, model)

        # end

        post_experiment(args, model)



if __name__ == '__main__':
    # op_2_1 = {'pooling': 'avg-token', 'mask_mode': 'cross-wise', 'additional_ban': 2}
    # op_3_1 = {
    #     "train_path": './data/aida/IO/train_samples',
    #     "t2d_path": './data/aida/IO/test_samples_t2d',
    #     "limaye_path": './data/aida/IO/test_samples_limaye',
    #     "wiki_path": './data/aida/IO/test_samples_wikipedia',
    #     "epochs_num": 30,
    # }
    #
    # for ds_options in [op_3_1]:
    #     for key_options in [op_2_1]: # process_2
    #         predefined_dict_groups = {
    #                                   'debug_options':{
    #                                       'logger_dir_name':'./col_cls_workspace/log_tem',
    #                                       'logger_file_name':'./col_cls_workspace/rec_all_tem'
    #                                   },
    #                                   'key_set_group':key_options,
    #                                   'ds_set_group':ds_options
    #                                   }
    #         print(predefined_dict_groups)
    #         experiment(repeat_time=1, predefined_dict_groups=predefined_dict_groups)
    pass