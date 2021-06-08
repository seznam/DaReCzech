import os
import argparse
from torch.utils.data import DataLoader
import torch
from transformers import ElectraTokenizerFast
from pytorch_finetuning import pytorch_finetuning
from pytorch_finetuning.electra_for_logistic_regression import ElectraForLogisticRegression

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("tokenizer", type=str, help="Path to folder storing tokenizer")
    parser.add_argument("train_tsv", type=str, help="Path to TSV file with training data")
    parser.add_argument("dev_tsv", type=str, help="Path to TSV file with development data")
    parser.add_argument("save_path", type=str, help="Path to folder to save checkpoints")
    parser.add_argument("pretrained_electra_path", help="Path to pretrained Electra model (TODO make this optional)") 

    parser.add_argument("--doc_max_len", default=128, help="Max number of tokens to use (do not use more than 512 tokens")
    parser.add_argument("--gpu_num", default="0", help="GPU ID")
    parser.add_argument("--random_seed", default=0, help="Random seed")
    parser.add_argument("--num_epochs", default=20, help="Number of training epochs")

    args = parser.parse_args()    

    TOKENIZER_PATH=args.tokenizer
    PRETRAINED_ELECTRA_PATH=args.pretrained_electra_path
    SAVE_PATH=args.save_path
    TRAIN_TSV=args.train_tsv
    DEV_TSV=args.dev_tsv
    DOC_MAX_LEN=args.doc_max_len
    NUM_TRAIN_EPOCHS=args.num_epochs

    tokenizer = ElectraTokenizerFast(TOKENIZER_PATH, strip_accents=False)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_num}")
    else:
        device = torch.device("cpu")

    train_dataset = pytorch_finetuning.RelevanceDataset(
        TRAIN_TSV,
        max_len=DOC_MAX_LEN,
        tokenizer=tokenizer,
        nrows=None,
    )
    dev_dataset = pytorch_finetuning.RelevanceDataset(
        DEV_TSV,
        max_len=DOC_MAX_LEN,
        tokenizer=tokenizer,
        nrows=None,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=5, pin_memory=False, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, num_workers=5, pin_memory=False)

    metrics = {
        "p_at_10": lambda model, predictions: pytorch_finetuning.get_p_at_10_precision(None, None, predictions.squeeze(-1), 
                dev_dataset.get_column('label'), dev_dataset.get_column('query'))
    }

    model_name=f"querydoc_electra-{args.random_seed}"

    ## MAIN
    # Load pre-trained model, fine-tune on TRAIN_TSV data and save 
    # P@10 progression can be inspected using Tensorboard
    pytorch_finetuning.train(
        PRETRAINED_ELECTRA_PATH,
        train_loader, 
        dev_loader, 
        num_epochs=NUM_TRAIN_EPOCHS,
        device=device,
        model_class=ElectraForLogisticRegression,
        saving_path=SAVE_PATH,
        finetuning_model_name=model_name,
        metrics=metrics,
        random_seed=args.random_seed,
        grad_acc_steps=8,
    )   
