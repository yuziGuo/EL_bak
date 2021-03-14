from col_cls_workspace.task_col_classifier import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int)
args = parser.parse_args()

if __name__ == '__main__':
    op_2_1 = {'pooling': 'avg-cell-seg', # avg-token, 'avg-cell-seg'
              'mask_mode': 'cross-wise',
              'shuffle_rows': True,
              'additional_ban': 3,
              'row_wise_fill': False,
              'has_high_level_cls': True}
    op_3_1 = {
        "train_path": './data/aida/IO/train_samples',
        "t2d_path": './data/aida/IO/test_samples_t2d',
        "limaye_path": './data/aida/IO/test_samples_limaye',
        "wiki_path": './data/aida/IO/test_samples_wikipedia',
        "epochs_num": 30,
    }

    for ds_options in [op_3_1]:
        for key_options in [op_2_1]: # process_2
            predefined_dict_groups = {
                                      'debug_options': {
                                          'logger_dir_name': './col_cls_workspace/log_tem_feb',
                                          'logger_file_name': './col_cls_workspace/rec_all_feb',
                                          'tx_logger_dir_name': './col_cls_workspace/runs_feb/',
                                          'exp_name':  "retain-high-level-cls+shuffle+avg-cell-seg+segpos-embedding-1"+"-"+str(args.cuda),
                                          'info': '''
        pos_token = torch.arange(0, word_emb.size(1)).unsqueeze(0).repeat(word_emb.size(0),1).long().to(word_emb.device)
        pos_emb_token = self.position_embedding(pos_token)
        pos_emb = pos_emb_token
        
        ones = torch.ones_like(seg)
        twos = ones * 2
        seg_token = torch.where(seg % 10000 // 100 > 1, twos, seg)
        seg_token = torch.where(seg % 10000 // 100 == 1, ones, seg_token)
        seg_emb = self.segment_embedding(seg_token)
        
        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
                                          '''
                                      },
                                      'key_set_group': key_options,
                                      'ds_set_group': ds_options
                                      }
            print(predefined_dict_groups)
            experiment(repeat_time=6, predefined_dict_groups=predefined_dict_groups)
