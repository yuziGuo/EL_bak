#!/bin/bash
DATA_PATH="aida_4_cols"
WEB_PRE_MODEL="./models/bert_model.bin-000"
# WEB_PRE_MODEL="./models/tabert.bin-999"
# WEB_PRE_MODEL="./models/bert_plus_webtable_0214.bin-200000"
# WEB_PRE_MODEL="/data/gyh/models/bert_plus_webtable_0207-2300.bin-200000"
PY_SCRIPT_2="run_table_encoder_yuhe.py"

FINETUNED_MODEL="TEMP-MODEL-1023"
#for i in 1 2 3 4 5 6 7 8 9 10
for i in 1
do
CUDA_VISIBLE_DEVICES=1,2 \
 python ${PY_SCRIPT_2} \
 --pretrained_model_path ${WEB_PRE_MODEL} \
 --vocab_path models/google_uncased_en_vocab.txt  \
 --train_path /home/gyh/pack_for_debug_2/UER-py-master/ff_train_samples \
 --not_train False \
 --dev_path /home/gyh/pack_for_debug_2/UER-py-master/ff_test_samples_t2d \
 --test_path /home/gyh/pack_for_debug_2/UER-py-master/ff_test_samples_limaye \
 --epochs_num 20 --epochs_num_namely 8  --batch_size 32 --seq_length 80 \
 --encoder bert --mask_mode crosswise --taskname sur \
 --output_model_path ${FINETUNED_MODEL}
done


