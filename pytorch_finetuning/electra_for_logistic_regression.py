import logging
from typing import List, Union

import torch
from torch import nn
from transformers.activations import get_activation
from transformers.file_utils import ModelOutput
from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel

logger = logging.getLogger(__name__)


class ElectraCustomLogisticRegressionHead(nn.Module):
    """Head for logistic regression tasks."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        num_labels: int,
        add_hidden: bool = True,
        output_activation: torch.nn.Module = torch.nn.Sigmoid,
    ):
        super().__init__()
        self.add_hidden = add_hidden

        self.dropout = nn.Dropout(hidden_dropout_prob)

        if self.add_hidden:
            self.dense = nn.Linear(input_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, num_labels)
        else:
            self.out_proj = nn.Linear(input_size, num_labels)

        self.output_activation = output_activation

    def forward(self, features, **kwargs):
        #         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])

        x = self.dropout(features)
        if self.add_hidden:
            x = self.dense(x)
            x = get_activation("gelu")(x)
            # used gelu here
            x = self.dropout(x)

        x = self.out_proj(x)
        x = self.output_activation()(x)
        return x


class ElectraForLogisticRegression(ElectraPreTrainedModel):
    """ELECTRA Model with a logistic regression on top of fully connected"""

    def __init__(
        self,
        config,
        electra_hidden_size: int = 256,
        additional_dense_hidden_size: int = 256,
        hidden_dropout_prob: float = 0.1,
        add_hidden: bool = True,
        output_activation: torch.nn.Module = torch.nn.Sigmoid,
    ):
        super().__init__(config)
        self.classifier = ElectraCustomLogisticRegressionHead(
            input_size=electra_hidden_size,
            hidden_size=additional_dense_hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            num_labels=1,
            add_hidden=add_hidden,
            output_activation=output_activation,
        )
        self.init_weights()

        self.electra = ElectraModel(config)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        return_dict=False,
    ):
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions=output_attentions,
        )
        attn_state = discriminator_hidden_states[-1]
        electra_last_hidden_states = discriminator_hidden_states[0]
        electra_output_for_cls_token = electra_last_hidden_states[:, 0, :]
        outputs = self.classifier(electra_output_for_cls_token)
        out = {"logits": outputs}

        loss = None

        if labels is not None:
            #  We are doing regression
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(outputs.view(-1), labels.view(-1))
            out["loss"] = loss

        if return_dict:
            if output_attentions:
                out["attentions"] = attn_state
            return ModelOutput(out)
        else:
            if output_attentions:
                if loss is not None:
                    return loss, outputs, attn_state
                else:
                    return outputs, output_attentions, attn_state
            else:
                if loss is not None:
                    return loss, outputs
                else:
                    return [outputs]


class ElectraRelevanceTeacher(nn.Module):
    """
    ELECTRA Model transformer teacher based on ELECTRA Model transformer
    with a logistic regression on top of fully connected
    """

    def __init__(
        self,
        model_class: torch.nn.Module,
        pytorch_dumps: Union[List[str], str],
        cross_model_distillation: bool,
        model_kwargs,
        eval_mode: bool = True,
    ):
        super().__init__()
        if isinstance(pytorch_dumps, str):
            pytorch_dumps = [pytorch_dumps]
        if isinstance(pytorch_dumps, list):
            self.teacher = [
                model_class.from_pretrained(
                    pytorch_dump,
                    output_activation=(
                        torch.nn.Sigmoid
                        if cross_model_distillation else torch.nn.Tanh
                    ),
                    **(model_kwargs or {}),
                )
                for pytorch_dump in pytorch_dumps
            ]
            if not eval_mode:
                for teacher in self.teacher:
                    teacher.train()
        else:
            raise TypeError(
                "starting_pytorch_dumps must be of a type str or List[str]"
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        return_dict=False,
        student_n_layers=None,
    ):
        """
        :param student_n_layers: A number of hidden layers in a student model.
        This paramere is optional and is required only if loss over attention
        states between a student and a teacher is computed.
        """
        teacher_outs = [
            teacher(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                output_attentions=output_attentions,
                return_dict=True,
            )
            for teacher in self.teacher
        ]

        teacher_gold_logits = self._get_gold_logits(teacher_outs)
        out = {"logits": teacher_gold_logits}
        if output_attentions:
            teacher_gold_attentions = self._get_teacher_attentions(
                teacher_outs, student_n_layers
            )
            out["attentions"] = teacher_gold_attentions

        if return_dict:
            return ModelOutput(out)
        else:
            if output_attentions:
                return teacher_gold_logits, teacher_gold_attentions
            else:
                return [teacher_gold_logits]

    def _get_gold_logits(
        self,
        teacher_outs: List[ModelOutput],
    ) -> torch.Tensor:
        """
        This function returns logit predictions of a teacher, or a mean
        logit predictions when multiple teachers are used.
        """
        return torch.stack(
            [teacher_outs[n].logits for n in range(len(teacher_outs))], 0
        ).mean(0)

    def _get_teacher_attentions(
        self,
        teacher_outs: List[ModelOutput],
        student_n_layers: int,
    ) -> Union[List[tuple], tuple]:
        """
        This function returns either selected attention layers when a single
        teacher is provided, or mean of selected attention layers of multiple
        teachers if an ensemble teacher is provided.

        :param student_n_layers: A number of hidden layers in a student model.
        """
        # Default settings of layers to be taken for knowledge distillation
        # TODO: make layers chosen for distillation to be an input parameter
        LAYERS_DICT = {3: [0, 6, 11], 4: [0, 3, 8, 11], 6: [0, 2, 4, 7, 9, 11]}
        layers = LAYERS_DICT[student_n_layers]

        if isinstance(teacher_outs, list):
            return tuple(
                torch.stack(
                    [
                        teacher_outs[n].attentions[i]
                        for n in range(len(teacher_outs))
                    ],
                    0
                ).mean(0)
                for i in layers
            )
        else:
            error_msg = (
                "Teacher output must be a type of ModelOutput or "
                f"List[ModelOutput] but it is a type of {type(teacher_outs)}."
            )
            raise TypeError(error_msg)
