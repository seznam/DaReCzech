import argparse
import logging

import torch
from torch.utils.data import DataLoader
from transformers import ElectraTokenizerFast

from pytorch_finetuning.electra_for_logistic_regression import \
    ElectraForLogisticRegression
from pytorch_finetuning.siamese_electra import \
    SiameseElectraWithResidualMaxWithAdditionalHiddenLayer
from pytorch_finetuning import pytorch_finetuning

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.DEBUG,
    datefmt="%I:%M:%S",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help="Path to folder with saved model"
    )
    parser.add_argument(
        "eval_tsv", type=str, help="Path to TSV file with training data"
    )

    parser.add_argument(
        "--is_querydoc",
        action="store_true",
        default=False,
        help="Evaluate query-doc model.",
    )
    parser.add_argument(
        "--is_siamese",
        action="store_true",
        default=False,
        help="Evaluate siamese model.",
    )

    parser.add_argument(
        "--gpu_num", default="0", help="GPU ID, Use -1 to run on CPU"
    )
    parser.add_argument(
        "--doc_max_len",
        default=128,
        help="Max number of tokens to use (do not use more than 512 tokens",
    )

    args = parser.parse_args()

    if not args.is_querydoc and not args.is_siamese:
        raise ValueError("Either --is_querydoc or --is_siamese must be set!")

    tokenizer = ElectraTokenizerFast.from_pretrained("Seznam/small-e-czech")

    if torch.cuda.is_available() and args.gpu_num != "-1":
        device = torch.device(f"cuda:{args.gpu_num}")
    else:
        device = torch.device("cpu")

    dataset_cls = (
        pytorch_finetuning.RelevanceDataset
        if args.is_querydoc
        else pytorch_finetuning.SiameseRelevanceDataset
    )
    dataset = dataset_cls(
        args.eval_tsv,
        max_len=args.doc_max_len,
        tokenizer=tokenizer,
        nrows=None,
    )

    dataset_loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=5,
        pin_memory=False
    )

    metrics = {
        "p_at_10":
        lambda model, predictions: pytorch_finetuning.get_p_at_10_precision(
            None,
            None,
            predictions.squeeze(-1),
            dataset.get_column("label"),
            dataset.get_column("query"),
        )
    }

    model_cls = (
        ElectraForLogisticRegression
        if args.is_querydoc
        else SiameseElectraWithResidualMaxWithAdditionalHiddenLayer
    )
    model = model_cls.from_pretrained(args.model)
    model.to(device)

    predictions = pytorch_finetuning.get_predictions(
        model, dataset_loader, device
    )

    for metric_name, metric_func in metrics.items():
        print(f"{metric_name}: {metric_func(model, predictions)}")
