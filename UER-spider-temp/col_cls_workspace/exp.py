from col_cls_workspace.task_col_classifier import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int)
args = parser.parse_args()

if __name__ == '__main__':
    op_2_1 = {
        'high_level_clses' : [
            'TAB'
            # 'COL'
            # 'CELL'
                              ],  # ['TAB', 'COL', 'CELL', 'ROW', 'NL']
        'option' : {
            # 'tab': 'tab-cls'
            # 'first-column': 'avg-cell-cls',
            'first-row': 'avg-token'
            # 'first-cell': 'avg-col-wise'
            # 'first-column': 'col-cls'
            # 'first-column': 'avg-token'
            },
        'mask_mode' : 'cross-wise',  # in ['row_wise', 'col_wise', 'cross_wise', 'cross_and_hier_wise']
        'embedding' : 'bert',
        
        'noise_num' : 2,
        'seq_len' : 128,
        'batch_size': 32
    }
    op_3_1 = {
        'row_wise_fill': False,
        'shuffle_rows': False,
        'moving_window_for_micro_table': True,
        "epochs_num": 3,

        "train_path": './data/aida/IO/train_samples',
        "t2d_path": './data/aida/IO/test_samples_t2d',
        "limaye_path": './data/aida/IO/test_samples_limaye',
        "wiki_path": './data/aida/IO/test_samples_wikipedia'
    }

    for ds_options in [op_3_1]:
        for key_options in [op_2_1]: # process_2
            predefined_dict_groups = {
                                      'debug_options': {
                                          'logger_dir_name': './col_cls_workspace/log2',
                                          'logger_file_name': './col_cls_workspace/rec_all2',
                                          'tx_logger_dir_name': './col_cls_workspace/runs2/',
                                          'exp_name':  "nosche-unshuffle-bz32-sl128+window(sample-wise-test)+2e5+2noise+first-row+bertpos"+"-"+str(args.cuda),
                                      },
                                      'key_set_group': key_options,
                                      'ds_set_group': ds_options
                                      }
            print(predefined_dict_groups)
            experiment(repeat_time=5, predefined_dict_groups=predefined_dict_groups)
