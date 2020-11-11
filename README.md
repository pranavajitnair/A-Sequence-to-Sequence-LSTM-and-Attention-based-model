# A Sequence to Sequence LSTM and Attention based model
The model could be used for any sequence-to-sequence task, my modifying utils.py. This particular model( i.e utils.py) is for generating expplanations for [CommonsenseQA](https://arxiv.org/pdf/1811.00937.pdf) dataset. 

The opened ended explanations( and spans in the questions which may be indicative of the correct answer) have been generated as a part of [Explain Yourself!
Leveraging Language Models for Commonsense Reasoning](https://arxiv.org/pdf/1906.02361.pdf).

The authors report a baseline result of 4.1 BLEU using GPT and also show that using these explanations for CommonsenseQA significanlty improves the accuracy.
The result obtained by this model is 4.137 BLEU. The model overfits on the data( due its small size) even with high dropout rates, drop connect and label smoothing.

To train and evaluate the model run
```
  python3 train.py
```
Optional arguments
```
  --lr                learning rate
  --input_size        embedding size
  --hidden_size       hidden size of LSTM
  --dev_com_path      path to decvelopment file for CommonsenseQA
  --train_com_path    path to training file for CommonsenseQA
  --dev_cose_path     path to COS-E development file
  --train_cose_path   path to COS-E training file
  --use_pretrained    whether to use a pretrained model
  --pretrained_path   path to the pretrained model
  --iters             number of training iterations
  --bs                batch size
  --max_norm          maximum gradient norm for parameters
  --min_decode_len    minimum decoding length
  --max_decode_len    maximum decoding length
  --beam_size         beam size
```
