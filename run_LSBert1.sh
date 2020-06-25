export BERT_DIR=/home/qiang/Desktop/pytorch-pretrained-BERT/
export Result_DIR=/home/qiang/Desktop/pytorch-pretrained-BERT/results


python3 LSBert1.py \
  --do_eval \
  --do_lower_case \
  --num_selections 10 \
  --eval_dir $BERT_DIR/datasets/BenchLS.txt \
  --bert_model bert-large-uncased-whole-word-masking \
  --max_seq_length 250 \
  --word_embeddings /media/qiang/ee63f41d-4004-44fe-bcfd-522df9f2eee8/wikipedia/fastText/crawl-300d-2M-subword.vec\
  --word_frequency $BERT_DIR/frequency_merge_wiki_child.txt \
  --output_SR_file $Result_DIR/aaa




   ##lex.mturk.txt \
