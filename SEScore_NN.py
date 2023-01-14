import evaluate
import torch
from transformers import AutoModel, AutoTokenizer
from comet.models.regression.referenceless import ReferencelessRegression
from typing import Dict
import datasets

import comet
from typing import Dict
import torch.nn as nn
import torch
from comet.encoders.base import Encoder
from comet.encoders.bert import BERTEncoder
from transformers import AutoModel, AutoTokenizer


class robertaEncoder(BERTEncoder):
    def __init__(self, pretrained_model: str) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModel.from_pretrained(
            pretrained_model, add_pooling_layer=False
        )
        self.model.encoder.output_hidden_states = True

    @classmethod
    def from_pretrained(cls, pretrained_model: str) -> Encoder:
        return robertaEncoder(pretrained_model)

    def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        last_hidden_states, _, all_layers = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }


class OurSEScore(ReferencelessRegression):
    def __init__(self, pretrained_model='xlm-roberta-large'):
        super(ReferencelessRegression, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.estimator = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.15, inplace=False),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.15, inplace=False),
            nn.Linear(in_features=1024, out_features=1, bias=True),
        )

    def compute_2(self, references, predictions):
        src = self.tokenizer(references).data
        mt = self.tokenizer(predictions).data
        return self.forward(torch.Tensor(src['input_ids']).long(), torch.Tensor(src['attention_mask']),
                            torch.Tensor(mt['input_ids']).long(), torch.Tensor(mt['attention_mask']))


score = OurSEScore()
res = score.compute_2(
    references=['sescore is a simple but effective next-generation text evaluation metric'],
    predictions=['sescore is simple effective text evaluation metric for next generation']
)
