# DaReCzech Dataset

DaReCzech is a **dataset for text relevance ranking in Czech**. The dataset consists of more than 1.6M annotated query-documents pairs, which makes it one of the largest available datasets for this task.

The dataset was introduced in paper [Siamese BERT-based Model for Web Search Relevance Ranking Evaluated on a New Czech Dataset](https://arxiv.org/pdf/2112.01810.pdf) which has been accepted at the IAAI 2022 (Innovative Application Award).

## Obtaining the Annotated Data
Please, first read a [disclaimer](disclaimer.md) that contains the terms of use. If you comply with them, send an email to srch.vyzkum@firma.seznam.cz and the link to the dataset will be sent to you. 

## Overview 
DaReCzech is divided into four parts: 
- Train-big (more than 1.4M records) – intended for training of a (neural) text relevance model
- Train-small (97k records) – intended for GBRT training (with a text relevance feature trained on Train-big)
- Dev (41k records)
- Test (64k records)

Each set is distributed as a .tsv file with 6 columns:
- ID – unique record ID
- query – user query
- url – URL of annotated document
- doc – representation of the document under the URL, each document is represented using its title, URL and Body Text Extract (BTE) that was obtained using the internal module of our search engine
- title: document title
- label – the annotated relevance of the document to the query. There are 5 relevance labels ranging from 0 (the document is not useful for given query) to 1 (document is for given query useful)

The files are UTF-8 encoded. The values never contain a tab and are not quoted nor escaped – to load the dataset in pandas, use
```
import csv
import pandas as pd
pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE)
```

## Baselines
We provide code to train two BERT-based baseline models: a query-doc model ([train_querydoc_model.py](train_querydoc_model.py)) and a siamese model ([train_siamese_model.py](train_siamese_model.py)).

Before running the scripts, install requirements that are listed in [requirements.txt](requirements.txt). The scripts were tested with Python 3.6.

```
pip install -r requirements.txt
```

### Model Training

To train a query-doc model with default settings, run:

```
python train_querydoc_model.py train_big.tsv dev.tsv outputs
```

To train a siamese model without a teacher, run:
```
python train_siamese_model.py train_big.tsv dev.tsv outputs
```

To train a siamese model with a trained query-doc teacher, run:
```
python train_siamese_model.py train_big.tsv dev.tsv outputs --teacher path_to_query_doc_checkpoint
```

Note that example scripts run training with our (unsupervisedly) pretrained [Small-E-Czech](https://huggingface.co/Seznam/small-e-czech) model.

### Model Evaluation

To evaluate the trained query-doc model on test data, run:
```
python evaluate_model.py model_path test.tsv --is_querydoc
```

To evaluate the trained siamese model on test data, run:
```
python evaluate_model.py model_path test.tsv --is_siamese
```

## Acknowledgements

If you use the dataset in your work, please cite the original paper:

```
@article{Kocián_Náplava_Štancl_Kadlec_2022, title={Siamese BERT-Based Model for Web Search Relevance Ranking Evaluated on a New Czech Dataset}, volume={36}, url={https://ojs.aaai.org/index.php/AAAI/article/view/21502}, DOI={10.1609/aaai.v36i11.21502}, number={11}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Kocián, Matěj and Náplava, Jakub and Štancl, Daniel and Kadlec, Vladimír}, year={2022}, month={Jun.}, pages={12369-12377} }
```
