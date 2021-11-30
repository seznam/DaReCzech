import argparse
import logging

import torch
from torch.utils.data import DataLoader
from transformers import ElectraTokenizerFast

from pytorch_finetuning.electra_for_logistic_regression import \
    ElectraForLogisticRegression
from pytorch_finetuning import pytorch_finetuning

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.DEBUG,
    datefmt="%I:%M:%S",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_tsv", type=str, help="Path to TSV file with training data"
    )
    parser.add_argument(
        "dev_tsv", type=str, help="Path to TSV file with development data"
    )
    parser.add_argument(
        "save_path", type=str, help="Path to folder to save checkpoints"
    )

    parser.add_argument(
        "--doc_max_len",
        default=128,
        help="Max number of tokens to use (do not use more than 512 tokens",
    )
    parser.add_argument("--batch_size", default=32, help="Batch size")
    parser.add_argument(
        "--grad_acc_steps", default=8, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--gpu_num", default="0", help="GPU ID, Use -1 to run on CPU"
    )
    parser.add_argument(
        "--random_seed", default=0, help="Random seed"
    )
    parser.add_argument(
        "--num_epochs", default=20, help="Number of training epochs"
    )
    args = parser.parse_args()

    tokenizer = ElectraTokenizerFast.from_pretrained("Seznam/small-e-czech")

    if torch.cuda.is_available() and args.gpu_num != "-1":
        device = torch.device(f"cuda:{args.gpu_num}")
    else:
        device = torch.device("cpu")

    train_dataset = pytorch_finetuning.RelevanceDataset(
        args.train_tsv,
        max_len=args.doc_max_len,
        tokenizer=tokenizer,
        nrows=None,
    )
    dev_dataset = pytorch_finetuning.RelevanceDataset(
        args.dev_tsv,
        max_len=args.doc_max_len,
        tokenizer=tokenizer,
        nrows=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=5,
        pin_memory=False,
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        num_workers=5,
        pin_memory=False
    )

    metrics = {
        "p_at_10": lambda model, predictions:
        pytorch_finetuning.get_p_at_10_precision(
            None,
            None,
            predictions.squeeze(-1),
            dev_dataset.get_column("label"),
            dev_dataset.get_column("query"),
        )
    }

    model_name = f"querydoc_electra-{args.random_seed}"

    # MAIN
    # Load pre-trained model, fine-tune on TRAIN_TSV data and save
    # P@10 progression can be inspected using Tensorboard
    pytorch_finetuning.train(
        "Seznam/small-e-czech",
        train_loader,
        dev_loader,
        num_epochs=args.num_epochs,
        device=device,
        model_class=ElectraForLogisticRegression,
        saving_path=args.save_path,
        finetuning_model_name=model_name,
        metrics=metrics,
        random_seed=args.random_seed,
        grad_acc_steps=args.grad_acc_steps,
    )
