# # A Simple BERT-Based Approach for Lexical Simplification
   Lexical simplification (LS) aims to replace complex words in a given sentence with their simpler alternatives of equivalent meaning. Recently unsupervised lexical simplification approaches only rely on the complex word itself regardless of the given sentence to generate candidate substitutions, which will inevitably produce a large number of spurious candidates. We present a simple BERT-based LS approach that makes use of the pre-trained unsupervised deep bidirectional representations BERT. We feed the given sentence masked the complex word into the masking language model of BERT to generate candidate substitutions. By considering the whole sentence, the generated simpler alternatives are easier to hold cohesion and coherence of a sentence. Experimental results show that our approach obtains obvious improvement on standard LS benchmark.
   

## Pre-trained models

- [FastText](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl300d-2M-subword.zip) (word embeddings trained using FastText)
- [BERT based on Pytroch](https://github.com/huggingface/pytorch-transformers)

## How to run this code

The project is based on Python 3.5

(1) Download the code of BERT based on Pytorch. In our experiments, we adopted pretrained [BERT-Large, Uncased (Whole Word Masking)](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip).

(2) Copy the files provided by the project into the main file of BERT

(3) download the pre-trained word embeddings using FastText.

(4) run "./run_LS_BERT.sh".

## Idea




## Citation

[BERT-LS technical report](https://arxiv.org/pdf/1907.06226.pdf)

```
@article{qiang2018STTP,
  title =  {A Simple BERT-Based Approach for Lexical Simplification },
  author = {Qiang, Jipeng and 
            Li, Yun and
            Yi, Zhu and
            Yuan, Yunhao and 
            Wu, Xindong},
  journal = {arXiv preprint arXiv:1907.06226},
  year  =  {2019}
}


