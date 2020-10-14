# TerseBERT

This repository contains information and code for **TerseBERT**, a pretrained language model created by fine-tuning [BERT](https://github.com/google-research/bert). TerseBERT is not only able to predict which word is most likely in a given context (like a regular language model), but if any word is necessary at all. It was created as a component of a text simplification solution described in the article *[Multi-Word Lexical Simplification](https://www.aclweb.org/anthology/TODO.pdf)* presented at the [COLING 2020](https://coling2020.org/) conference in Barcelona.

For example, consider the sentence *The fat cat sat on the mat.* If we mask the word *mat* and ask for the most likely predictions, both BERT and TerseBERT suggest *floor*, *bed*, *table*, etc. If we mask the word *fat*, BERT proposes *black*, *white*, *big*, while TerseBERT offers the same predictions, but also reports a high probability (80%) of *[NONE]* token. This indicates the sentence is likely to have no words in the selected location, as we can simply say *The cat sat on the mat.*

This document is a guide for obtaining, training and using a TerseBERT model. If you need any more information consult [the paper](https://www.aclweb.org/anthology/TODO.pdf) or contact its authors! 

## Obtaining and using TerseBERT

The TerseBERT model trained for the study mentioned above is available in [Hugging Face Transformers](https://github.com/huggingface/transformers) format for download [here](http://homados.ipipan.waw.pl/tersebert-data/tersebert_pytorch_1_0.bin) (1.3 GB). Provided you have [PyTorch](https://pytorch.org/) and [NumPy](https://numpy.org/) installed, you can invoke TerseBERT in the following way:
```python
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertForMaskedLM

premodel='bert-large-uncased-whole-word-masking'
tokenizer = BertTokenizer.from_pretrained(premodel)
model_dict = torch.load("/PATH/TO/tersebert_pytorch_1_0.bin")
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=premodel, state_dict=model_dict)

sentence="The fat cat sat on the mat."
tokenized_text = tokenizer.tokenize(sentence)
masked_text=['[CLS]']+tokenized_text+['[SEP]']
masked_token=tokenized_text.index("fat")+1
masked_text[masked_token]='[MASK]'
indexed_tokens=tokenizer.convert_tokens_to_ids(masked_text)
segments_ids=[0]*len(masked_text)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
predictions = model(tokens_tensor, segments_tensors)
predicted_index = torch.argmax(predictions[0][0][masked_token]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
scores=predictions[0][0][masked_token].detach().numpy()
tops=(-scores).argsort()[0:10]
print(list(zip(tokenizer.convert_ids_to_tokens(tops),(np.exp(scores)/sum(np.exp(scores)))[tops])))
```
You should expect the following output:
```
[('[unused0]', 0.80043215), ('black', 0.057732496), ('white', 0.036740497), ('big', 0.011116397),
('little', 0.005187636), ('gray', 0.0025780434), ('fat', 0.0025481516), ('house', 0.0021788846),
('old', 0.0021117083), ('giant', 0.001906771)]
```
Note that Transformers is using the original BERT vocabulary (from *bert-large-uncased-whole-word-masking*), so the *[NONE]* token is displayed as *[unused0]*.

## Training your own TerseBERT 

To train your own TerseBERT, follow these steps:
1. Prepare a large corpus of documents for finetuning. We chose Wikipedia and used [WikiExtractor](https://github.com/attardi/wikiextractor) to extract plain text, but you can use any source, e.g. with domain-specific documents.
1. Choose a BERT model to fine-tune (we used *BERT-Large, Uncased (Whole Word Masking)*) and modify its dictionary to include the *[NONE]* token: see ```vocab_none.txt```.
1. Create pretraining data by using ```create_pretraining_data_none.py```, a variant of BERT's ```create_pretraining_data.py``` modified to insert *[NONE]* in random places in a defined number (we used 5%).
1. Run pretraining in the usual way (in our case 5000 steps was enough).
1. Convert the returned model [into Hugging Face format](https://huggingface.co/transformers/converting_tensorflow_models.html). 

## Licence

* Like the original BERT code, the modified pretraining script is licensed under [Apache Licence 2.0](http://www.apache.org/licenses/LICENSE-2.0).
* The pretrained model is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) licence.
