export Data_DIR=/home/qiang/Desktop/pytorch-pretrained-BERT/datasets
export Result_DIR=/home/qiang/Desktop/pytorch-pretrained-BERT/results


python LS_Bert.py \
  --do_eval \
  --do_lower_case \
  --num_selections 6 \
  --eval_dir $Data_DIR/BenchLS.txt \
  --bert_model bert-large-uncased-whole-word-masking \
  --max_seq_length 250 \
  --output_SR_file $Result_DIR/aaa




   ##lex.mturk.txt \