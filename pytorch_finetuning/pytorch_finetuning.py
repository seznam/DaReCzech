import csv
import logging
import os
import random
import socket
import time
from collections.abc import Iterable
from datetime import datetime as dt
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import transformers
from catboost.utils import eval_metric
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from .electra_for_logistic_regression import (ElectraForLogisticRegression,
                                              ElectraRelevanceTeacher)
from .progress_report import get_progress_reporter
from .siamese_electra import \
    SiameseElectraWithResidualMaxWithAdditionalHiddenLayer


class TextDataset(Dataset):
    def __init__(
        self, path, column, max_len, tokenizer, token_type_id, nrows=None
    ):
        self.path = path
        self.df = pd.read_csv(
            path,
            sep="\t",
            quoting=csv.QUOTE_NONE,
            nrows=nrows,
            usecols=[column, "label"],
        )

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.column = column
        self.token_type_id = token_type_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.at[index, self.column]
        label = self.df.at[index, "label"]

        input_dict = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation="longest_first",
        )

        input_ids = torch.tensor(input_dict["input_ids"])
        attention_mask = torch.tensor(input_dict["attention_mask"])
        token_type_ids = (
            torch.tensor(input_dict["attention_mask"]) * self.token_type_id
        )

        return input_ids, attention_mask, token_type_ids, label


class RelevanceDataset(Dataset):
    def __init__(self, path, max_len, tokenizer, nrows=None):
        self.path = path
        self.df = pd.read_csv(
            path, sep="\t", quoting=csv.QUOTE_NONE, nrows=nrows
        )

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        query = self.df.at[index, "query"]
        doc = self.df.at[index, "doc"]
        label = self.df.at[index, "label"]

        input_dict = self.tokenizer.encode_plus(
            # if we concatenate query and doc ourselves (by [SEP]),
            # token_type_ids will be all zero
            query,
            doc,
            max_length=self.max_len,
            padding="max_length",
            truncation="longest_first",
        )

        input_ids = torch.tensor(input_dict["input_ids"])
        attention_mask = torch.tensor(input_dict["attention_mask"])
        token_type_ids = torch.tensor(input_dict["token_type_ids"])

        return input_ids, attention_mask, token_type_ids, label

    def get_column(self, column_name):
        return self.df[column_name]


class SiameseRelevanceDataset(RelevanceDataset):
    def get_inputs(self, text, token_type_id):
        input_dict = self.tokenizer.encode_plus(
            # if we concatenate query and doc ourselves (by [SEP]),
            # token_type_ids will be all zero
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation="longest_first",
        )
        input_ids = torch.tensor(input_dict["input_ids"])
        attention_mask = torch.tensor(input_dict["attention_mask"])
        # mark query with token_type_id=0 and doc with token_type_id=1
        token_type_ids = (
            torch.tensor(input_dict["attention_mask"]) * token_type_id
        )
        return input_ids, attention_mask, token_type_ids

    def __getitem__(self, index):
        query = self.df.at[index, "query"]
        doc = self.df.at[index, "doc"]
        label = 2 * (self.df.at[index, "label"] - 0.5)

        result = []
        for i, text in enumerate([query, doc]):
            result.extend(self.get_inputs(text, int(i > 0)))
        result.append(label)
        return result


class SiameseRelevanceDatasetDistillation(RelevanceDataset):
    """
    A special dataset class used for training Siamese Electra with a teacher
    represented by a standard Electra model.
    """

    def get_inputs(self, text, token_type_id):
        input_dict = self.tokenizer.encode_plus(
            # if we concatenate query and doc ourselves (by [SEP]),
            # token_type_ids will be all zero
            text,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation="longest_first",
        )
        input_ids = torch.tensor(input_dict["input_ids"])
        attention_mask = torch.tensor(input_dict["attention_mask"])
        # mark query with token_type_id=0 and doc with token_type_id=1
        token_type_ids = (
            torch.tensor(input_dict["attention_mask"]) * token_type_id
        )
        return input_ids, attention_mask, token_type_ids

    def __getitem__(self, index):
        # Data for Siamese Electra
        query = self.df.at[index, "query"]
        doc = self.df.at[index, "doc"]
        label = self.df.at[index, "label"]

        result = []
        for i, text in enumerate([query, doc]):
            result.extend(self.get_inputs(text, int(i > 0)))
        result.append(label)

        # Data for Standard Electra
        result2 = []
        input_dict = self.tokenizer.encode_plus(
            # if we concatenate query and doc ourselves (by [SEP]),
            # token_type_ids will be all zero
            query,
            doc,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation="longest_first",
        )

        input_ids = torch.tensor(input_dict["input_ids"])
        attention_mask = torch.tensor(input_dict["attention_mask"])
        token_type_ids = torch.tensor(input_dict["token_type_ids"])
        result2.extend([input_ids, attention_mask, token_type_ids])
        result2.append(label)
        return result, result2


def _eval(model, dev_loader, device, metrics, writer, it=0, logger=None):
    logger = logger or logging
    logger.info("Evaluating")
    start_eval_time = time.time()
    model.eval()

    loss, predictions = get_predictions(
        model, dev_loader, device, compute_loss=True
    )
    # dev loss gets logged automatically
    writer.add_scalar("dev/Loss", loss, it)
    logger.info(f"dev/Loss: {loss}")
    if metrics:
        for name, metric in metrics.items():
            metric_val = metric(model, predictions)
            writer.add_scalar(name, metric_val, it)
            logger.info(f"{name}: {metric_val}")
    writer.flush()
    logger.info(
        "Evaluation took {:.2f} s".format(time.time() - start_eval_time)
    )


class Logger:
    """Logging through logging and to a file"""

    # this is probably doable using logging handlers only, but I could not make
    # it work (without getting duplicated messages etc.)
    def __init__(self, saving_path, file_logging_level=logging.INFO):
        self.saving_path = saving_path
        self.log_path = os.path.join(saving_path, "log.txt")
        self.file_logging_level = file_logging_level

    def write(self, text):
        with open(self.log_path, "a") as fout:
            print(dt.now(), text, file=fout)

    def debug(self, text):
        logging.debug(text)
        if self.file_logging_level <= logging.DEBUG:
            self.write(text)

    def info(self, text):
        logging.info(text)
        if self.file_logging_level <= logging.INFO:
            self.write(text)

    def warning(self, text):
        logging.warning(text)
        if self.file_logging_level <= logging.WARNING:
            self.write(text)


def get_model_iden(model_class):
    """Get model id in the form <model_type>_<time %Y_%m_%d_%H%M%S%f>[:-3]"""
    class_iden = {
        ElectraForLogisticRegression: "rel",
        SiameseElectraWithResidualMaxWithAdditionalHiddenLayer: (
            "residual_max_with_hidden"
        ),
    }.get(model_class, "unk")

    return "{}_{}".format(
        class_iden,
        dt.strftime(dt.now(), "%Y_%m_%d_%H%M%S%f")[:-3]
    )


def _get_dataset_path(data_loader):
    try:
        return data_loader.dataset.path
    except Exception:
        return None


def train(
    starting_pytorch_dump: Union[str, List[str]],
    train_loader: Union[torch.utils.data.DataLoader, Iterable],
    dev_loader: Union[torch.utils.data.DataLoader, Iterable],
    num_epochs: int,
    device: torch.device,
    finetuning_model_name: str,
    saving_path: str,
    random_seed: int = 0,
    model_class=ElectraForLogisticRegression,
    loss: torch.nn.Module = None,
    metrics=None,
    model_kwargs=None,
    learning_rate: int = 5e-5,
    grad_acc_steps: int = 1,
    eval_initial_model: bool = True,
    student_starting_pytorch_dump: str = None,
    attn_loss_distil: bool = False,
    notebook: bool = False,
    progress_reporting: str = "tqdm",
):
    """
    Function for the training of transformer models.
    Currently, both standard training and training utilizing knowledge
    distillation (i.e. teacher -> student learning) is supported.
    Distillation training is automatically used once both
    `starting_pytorch_dump` and `student_starting_pytorch_dump` are provided.

    Input:
    -----
    starting_pytorch_dump: `Union[str, List[str]]` - An absolute path
    to the folder with the following files
    [pytorch_model.bin, config.json, vocab.txt].
    By default, when a standard training is run, i.e. knowledge distillation
    is not used, this parameter should be `str` leading to a single model.
    When knowledge distillation is used, this parameter can be either either
    `str` or `List[str]` depending on whether a single teacher or an ensemble
    one is used.

    student_starting_pytorch_dump: `str` - An absolute path to the folder
    with the following files [pytorch_model.bin, config.json, vocab.txt].

    attn_loss_distil: `bool` - An indication whether the MSE loss between
    the student's and teacher's attention hidden states should be used
    for transformer-layer distillation.

    notebook: `bool` - Indication whether the function is used in a notebook
    or in a command line. This has an effect on tqdm functionality.

    progress_reporting: "tqdm", "eta" or None. Which type of progress reporting
    to use.

    """
    if not student_starting_pytorch_dump and (
        isinstance(starting_pytorch_dump, list)
    ):
        raise ValueError(
            "When student is not used, only a single model for training "
            "must be provided"
        )

    distillation = starting_pytorch_dump and student_starting_pytorch_dump

    # realize whether we do cross_model_distillation or standard KD
    if not distillation or (
        student_starting_pytorch_dump
        and model_class is ElectraForLogisticRegression
    ):
        # Standard knowledge distillation
        cross_model_distillation = False
    else:
        # Cross-model distillation from ElectraForLogisticRegression
        cross_model_distillation = True

    os.makedirs(saving_path)

    logger = Logger(saving_path, file_logging_level=logging.DEBUG)

    logger.debug(socket.gethostname())
    logger.debug("transformers {}".format(transformers.__version__))
    logger.debug("torch {}".format(torch.__version__))
    logger.debug("random seed {}".format(random_seed))
    logger.debug("model_kwargs {}".format(model_kwargs))
    logger.debug("learning_rate {}".format(learning_rate))
    logger.debug("grad_acc_steps {}".format(grad_acc_steps))
    logger.debug("train set path {}".format(_get_dataset_path(train_loader)))
    logger.debug("dev set path {}".format(_get_dataset_path(dev_loader)))

    progress = get_progress_reporter(progress_reporting, logger)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    if isinstance(starting_pytorch_dump, (list, str)):
        if student_starting_pytorch_dump:
            # Teacher instantiation
            logger.info(
                f"""
            Instantiating ElectraForLogisticRegression from
            {starting_pytorch_dump}"""
            )
            # TODO: make model_class to be input arguments if ever needed
            teacher_model = ElectraRelevanceTeacher(
                model_class=ElectraForLogisticRegression,
                pytorch_dumps=starting_pytorch_dump,
                cross_model_distillation=cross_model_distillation,
                model_kwargs=model_kwargs,
            )
            # Student instantiation
            logger.info(
                f"""
            Instantiating student {model_class} from {starting_pytorch_dump}
            """
            )
            model = model_class.from_pretrained(
                student_starting_pytorch_dump, **(model_kwargs or {})
            )
            num_student_layers = model.config.num_hidden_layers
            logger.info(
                f"Number of student hidden layers: {num_student_layers}."
            )
        else:
            logger.info(
                f"Instantiating {model_class} from {starting_pytorch_dump}"
            )
            model = model_class.from_pretrained(
                starting_pytorch_dump, **(model_kwargs or {})
            )
    else:  # this is currently supported only for training without a teacher
        model = starting_pytorch_dump
        model_class = type(model)
    # before creating the optimizer https://pytorch.org/docs/stable/optim.html
    model.to(device)
    if distillation:
        for teacher in teacher_model.teacher:
            teacher.to(device)

    iden = get_model_iden(model_class)
    logger.info("identifier {}".format(iden))

    # initialize soft_loss & attention_loss if student provided
    if student_starting_pytorch_dump:
        soft_loss = torch.nn.MSELoss()
        attention_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(os.path.join("runs", finetuning_model_name) or None)
    it = 0

    if eval_initial_model:
        _eval(model, dev_loader, device, metrics, writer, it, logger=logger)

    optimizer.zero_grad()
    for epoch in progress(range(num_epochs), desc="training"):
        save_dir = os.path.join(saving_path, "epoch{}".format(epoch))
        os.makedirs(save_dir)

        model.train()
        logger.info("Epoch {}".format(epoch))
        for inputs in progress(train_loader, desc="epoch"):
            if not cross_model_distillation:
                # input_ids, attn_masks, token_type_ids, ..., labels
                inputs = inputs_to_cuda_and_no_double(inputs, device)
            else:
                # inputs represent the List[List[...]]
                inputs = [
                    inputs_to_cuda_and_no_double(input_i, device)
                    for input_i in inputs
                ]
            if not student_starting_pytorch_dump:  # = standard training
                _loss, *_ = model(*inputs)
            else:  # = distillation training
                _loss = _get_distil_loss(
                    model,
                    teacher_model,
                    inputs,
                    attn_loss_distil,
                    soft_loss,
                    attention_loss,
                    cross_model_distillation,
                )
            _loss /= grad_acc_steps  # loss normalization
            _loss.backward()

            if it % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            writer.add_scalar("Loss/train", _loss.item(), it)
            it += 1
        _eval(model, dev_loader, device, metrics, writer, it, logger=logger)

        model.save_pretrained(save_dir)
    writer.close()


def get_inputs(model, tokenizer, device, *args):
    input_dict = tokenizer.encode_plus(
        *args,
        max_length=model.config.max_position_embeddings,
        padding="max_length",
        truncation="longest_first",
    )
    input_ids = (
        torch.tensor(input_dict["input_ids"], device=device)
        .unsqueeze(0)
    )
    attention_mask = (
        torch.tensor(input_dict["attention_mask"], device=device)
        .unsqueeze(0)
    )
    token_type_ids = (
        torch.tensor(input_dict["token_type_ids"], device=device)
        .unsqueeze(0)
    )
    return input_ids, attention_mask, token_type_ids


def eval_model(model, tokenizer, device, *args):
    """Helper function evaluating model on one query-doc"""
    model.eval()
    model.to(device)
    inputs = get_inputs(model, tokenizer, device, *args)
    with torch.no_grad():
        output = model(*inputs)
    return output.detach().cpu().numpy()


def get_dev_loss_per_epoch(directory):
    dev_losses = []
    for d in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, d)):
            with open(os.path.join(directory, d, "dev_loss.txt")) as fin:
                dev_losses.append(float(fin.read().strip()))
    return dev_losses


def _get_distil_loss(
    model,
    teacher_model,
    inputs,
    attn_loss_distil,
    soft_loss_fn,
    attention_loss_fn,
    cross_model_distillation=False,
):
    """
    (Private) function returning distillation loss for a training utilsing
    knowledge distillation.

    Input:
    -----
    soft_loss_fn: Initialized instance of torch.nn.{loss_function}
    attention_loss_fn: Initialized instance of torch.nn.{loss_function}

    Output:
    ------
    loss: `torch.tensor([1,], dtype=torch.float)
    """
    student_outs = model(
        *inputs if not cross_model_distillation else inputs[0],
        output_attentions=attn_loss_distil,
        return_dict=True,
    )
    with torch.no_grad():
        teacher_outs = teacher_model(
            *inputs if not cross_model_distillation else inputs[1],
            output_attentions=attn_loss_distil,
            return_dict=True,
            student_n_layers=model.config.num_hidden_layers,
        )
    student_model_loss = student_outs.loss
    soft_loss = soft_loss_fn(
        student_outs.logits.view(-1), teacher_outs.logits.view(-1)
    )
    loss = 0.5 * (student_model_loss + soft_loss)

    if attn_loss_distil:
        attn_loss = torch.tensor(
            [
                attention_loss_fn(
                    student_outs.attentions[i], teacher_outs.attentions[i]
                )
                for i in range(model.config.num_hidden_layers)
            ]
        ).mean()
        loss += 0.5 * attn_loss
    return loss


def inputs_to_cuda_and_no_double(inputs, device):
    """
    Move all input Tensors from input to specified CUDA device
    and cast all torch.double (torch.float64) Tensors to
    torch.float (torch.float32) Tensors.
    """
    inputs = [
        x.cuda(device)
        if x.type() != "torch.DoubleTensor"
        else x.type(torch.FloatTensor).cuda(device)
        for x in inputs
    ]

    return inputs


def get_predictions(model, data_loader, device, compute_loss=False):
    """Get predictions for all inputs from a data loader"""
    model.eval()
    preds = []

    if compute_loss:
        sum_loss = 0

    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs_to_cuda_and_no_double(inputs, device)
            # don't input labels
            if compute_loss:
                loss, output, *_ = model(*inputs)
                sum_loss += loss.cpu() * len(inputs[0])
            else:
                output, *_ = model(*inputs[:-1])

            if len(output.shape) == 3:
                output = output[:, 0, :]

            preds.append(output.cpu())

    if compute_loss:
        return sum_loss / len(data_loader.dataset), np.concatenate(preds)
    else:
        return np.concatenate(preds)


def get_relevance_metric(metric, labels, predictions, group_ids):
    """Get catboost relevance metric value

    It is better to use this than catboost.utils.eval_metric as the latter
    likes to segfault when given incorrect args.

    :param metric: str  catboost metric identifier, e.g. "PrecisionAt:top=10"
    :param labels: [float]  labels in range [0, 1]
    :param predictions: [float]  predictions in range [0, 1]
    :param group_ids: can be a sequence of queries
    :return: float  metric value
    """
    assert len(labels) == len(predictions)
    assert len(predictions) == len(group_ids)
    assert np.max(labels) == 1
    assert np.min(labels) == 0

    # group_ids must be grouped
    df = (
        pd.DataFrame({
            "label": labels, "pred": predictions, "group_id": group_ids
        })
        .sort_values("group_id")
        .reset_index(drop=True)
    )

    return eval_metric(
        df["label"], df["pred"], group_id=df["group_id"], metric=metric
    )[0]


def get_p_at_10_precision(model, loader, predictions, labels, queries):
    return get_relevance_metric(
        "PrecisionAt:top={}".format(10),
        labels=labels,
        predictions=predictions,
        group_ids=queries,
    )
