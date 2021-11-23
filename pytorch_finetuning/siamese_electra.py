import logging
import os

import torch
import torch.nn.functional as F
from transformers.file_utils import ModelOutput
from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel


logger = logging.getLogger(__name__)


class CustomMetricPreTrainedModel(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, path):
        custom_metric = cls()
        custom_metric.load_state_dict(
            torch.load(os.path.join(path, "pytorch_model.bin"), torch.device("cpu")),
            strict=False,
        )
        return custom_metric


class CustomMetric(CustomMetricPreTrainedModel):
    def __init__(self):
        super().__init__()
        self.complex_features_nn = torch.nn.Linear(2 * 256, 3)
        self.out_proj = torch.nn.Linear(5, 1)

    def forward(self, embs, embs2):
        # torch.nn.CosineSimilarity export to onnx is not supported
        # cos_sim = torch.nn.CosineSimilarity()(embs, embs2)
        cos_sim = torch.reshape(
            torch.sum(
                torch.mul(
                    torch.nn.functional.normalize(embs, p=2, dim=1),
                    torch.nn.functional.normalize(embs2, p=2, dim=1),
                ),
                dim=1,
            ),
            (-1,),
        )

        euclidean_distance = torch.norm(embs - embs2, 2, dim=1)

        features = self.complex_features_nn(torch.cat((embs, embs2), 1))

        concatenated_embeddings = torch.cat(
            (cos_sim.view(-1, 1), euclidean_distance.view(-1, 1), features), 1
        )

        outputs = (torch.tanh(self.out_proj(concatenated_embeddings)),)
        return outputs


class ResidualMaxWithAdditionalHiddenLayer(CustomMetricPreTrainedModel):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(256, 512)
        self.layer_2 = torch.nn.Linear(512, 256)
        self.out_proj = torch.nn.Linear(258, 1)

    def forward(self, embs, embs2):
        # torch.nn.CosineSimilarity export to onnx is not supported
        # cos_sim = torch.nn.CosineSimilarity()(embs, embs2)
        cos_sim = torch.reshape(
            torch.sum(
                torch.mul(
                    torch.nn.functional.normalize(embs, p=2, dim=1),
                    torch.nn.functional.normalize(embs2, p=2, dim=1),
                ),
                dim=1,
            ),
            (-1,),
        )

        euclidean_distance = torch.norm(embs - embs2, 2, dim=1)

        max_emb = torch.max(embs, embs2)
        features_projected = F.dropout(torch.nn.GELU()(self.layer_1(max_emb)), 0.25)
        features = self.layer_2(features_projected)
        hidden = torch.nn.GELU()(features) + max_emb
        hidden = torch.cat(
            (cos_sim.view(-1, 1), euclidean_distance.view(-1, 1), hidden), 1
        )

        outputs = (torch.tanh(self.out_proj(hidden)),)
        return outputs


class SiameseElectraPretrainedModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.custom_metric = CustomMetric()
        self.init_weights()

        self.electra = ElectraModel(config)

    @staticmethod
    def return_model_output(outputs, labels, return_dict):
        out = {"logits": outputs[0]}
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(outputs[0].view(-1), labels.view(-1))
            outputs = [loss] + outputs
            out["loss"] = loss

        return ModelOutput(out) if return_dict else outputs


class SiameseElectraWithWeightedCLSPretrainedModel(SiameseElectraPretrainedModel):
    """
    Superclass for Siamese-like models utilizing weightening of [CLS] token.
    """

    def __init__(self, config):
        super().__init__(config)
        self.custom_metric = CustomMetric()
        self.init_weights()

        self.electra = ElectraModel(config)
        self.pooling = torch.nn.Conv1d(13, 1, 1)

    def get_weighted_cls_token(self, embs_hidden, embs2_hidden):
        """
        Args:
            embs_hidden: `Tuple[torch.tensor]` - Hidden states output by
            ElectraModel from transformers library.
            embs2_hidden: see `embs_hidden`
        """
        embs_cls = torch.cat([emb[:, 0, :].unsqueeze(1) for emb in embs_hidden], dim=1)
        embs2_cls = torch.cat(
            [emb[:, 0, :].unsqueeze(1) for emb in embs2_hidden], dim=1
        )
        # Make a linear combination of [CLS] tokens
        embs_pooled = self.pooling(embs_cls).squeeze(1)
        embs2_pooled = self.pooling(embs2_cls).squeeze(1)

        return embs_pooled, embs2_pooled


class SiameseElectraWithResidualMaxWithAdditionalHiddenLayer(
    SiameseElectraWithWeightedCLSPretrainedModel
):
    """
    Another version of SiameseElectraWithRedsidualMax extended by an additional
    hidden layer in a custom metric.
    """

    def __init__(self, config):
        super().__init__(config)
        self.custom_metric = ResidualMaxWithAdditionalHiddenLayer()
        self.init_weights()

        self.electra = ElectraModel(config)
        self.pooling = torch.nn.Conv1d(13, 1, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        input_ids2=None,
        attention_mask2=None,
        token_type_ids2=None,
        labels=None,
        output_attentions=False,
        return_dict=False,
    ):
        # run electra and take the embedding of [CLS] token
        # [0] selects last hidden states of Electra
        # [:, 0, :] indexes into [batch_size, max_seq_len, hidden_size]
        embs_hidden = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states
        embs2_hidden = self.electra(
            input_ids2,
            attention_mask2,
            token_type_ids2,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states

        embs_pooled, embs2_pooled = self.get_weighted_cls_token(
            embs_hidden, embs2_hidden
        )

        outputs = [self.custom_metric(embs_pooled, embs2_pooled)[0]]
        return self.return_model_output(outputs, labels, return_dict)
