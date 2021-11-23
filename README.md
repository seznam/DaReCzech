# DaReCzech Dataset

DaReCzech is a **dataset for text relevance ranking in Czech**. The dataset consists of more than 1.6 M annotated query-documents pairs, which makes it one of the largest available datasets for this task.

The dataset was introduced in paper TODO.

## Obtaining the Annotated Data
TODO

## Overview 
DaReCzech is divided into four parts: 
- Train-big (more than 1.4M records) -- intended for training of a (neural) text relevance model
- Train-small (97k records) -- intended for GBRT training
- Dev (41k records)
- Test (64k records)

Each set is distributed as a .tsv file with 6 columns:
- ID -- unique record ID
- query -- user query
- url -- URL of annotated document
- doc -- representation of the document under the URL, each document is represented using its title, URL and Body Text Extract (BTE) that was obtained using the internal module of our search engine
- title: document title
- label -- the annotated relevance between the query and document. There are 5 relevance labels ranging from 0 (the document is not useful for given query) to 1 (document is for given query useful)

Encoding: UTF-8.

## Baselines
We provide code to train two BERT-based baseline models: query-doc model ([train_querydoc_model.py](train_querydoc_model.py)) and siamese model ([train_siamese_model.py](train_siamese_model.py)). 

Before running the scripts, install requirements that are listed in [requirements.txt](requirements.txt).

```
pip install -r requirements.txt
```

To train a query-doc model with basic settings, run:

```
python train_querydoc_model.py train.tsv dev.tsv outputs
```

To train a siamese model without a teacher, run:
```
python train_siamese_model.py train.tsv dev.tsv outputs
```

To train it with a query-doc teacher, run:
```
python train_siamese_model.py train.tsv dev.tsv outputs --teacher path_to_doc_query_checkpoint
```

Note that example scripts run training with our (unsupervisedly) pretrained Small-E-Czech model ([https://huggingface.co/Seznam/small-e-czech](https://huggingface.co/Seznam/small-e-czech)).

## Acknowledgements

If you use the dataset in your work, please cite the original paper:

```

```