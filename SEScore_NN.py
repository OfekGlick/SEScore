import numpy as np
from comet.models.regression.referenceless import ReferencelessRegression

from typing import Dict
import torch.nn as nn
import torch
from comet.encoders.base import Encoder
from comet.encoders.bert import BERTEncoder
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


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
    def __init__(self,drop_out, pretrained_model='xlm-roberta-large'):
        super(ReferencelessRegression, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.Tanh(),
            nn.Dropout(p=drop_out, inplace=False),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.Tanh(),
            nn.Dropout(p=drop_out, inplace=False),
            nn.Linear(in_features=1024, out_features=1, bias=True),
        )


class CustomDataset(Dataset):
    def __init__(self, references, predictions, scores, pretrained_model='xlm-roberta-large'):
        self.references = references
        self.predictions = predictions
        self.scores = scores
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def __len__(self):
        return len(self.references)

    def __getitem__(self, idx):
        references_tokens = self.tokenizer(self.references[idx])
        references_input_ids = references_tokens.data['input_ids']
        references_attention_masks = references_tokens.data['attention_mask']
        predictions_tokens = self.tokenizer(self.predictions[idx])
        predictions_input_ids = predictions_tokens.data['input_ids']
        predictions_attention_masks = predictions_tokens.data['attention_mask']
        return references_input_ids, references_attention_masks, predictions_input_ids, predictions_attention_masks, \
               self.scores[idx]


def tokenize_and_pad(batch):
    references_input_ids, references_attention_masks, predictions_input_ids, predictions_attention_masks, scores = zip(
        *batch)
    references_input_ids = pad_sequence(
        [torch.Tensor(x).long() for x in references_input_ids], batch_first=True)
    references_attention_masks = pad_sequence(
        [torch.Tensor(x).long() for x in references_attention_masks], batch_first=True)
    predictions_input_ids = pad_sequence(
        [torch.Tensor(x).long() for x in predictions_input_ids], batch_first=True)
    predictions_attention_masks = pad_sequence(
        [torch.Tensor(x).long() for x in predictions_attention_masks], batch_first=True)
    scores = torch.Tensor(scores)
    return references_input_ids, references_attention_masks, predictions_input_ids, predictions_attention_masks, scores


def load_dataset(path):
    pass
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=1e-3)
    parser.add_argument('-batch_size',default = 16)
    parser.add_argument('-epochs',default = 1)
    parser.add_argument('-drop_out',default = 0.15)
    parser.add_argument('-beta_1',default = 0.9)
    parser.add_argument('-beta_2', default=0.99)
    parser.add_argument('-train_path',default = None)
    parser.add_argument('-test_path',default = None)
    args = parser.parse_args()
    return args
def train():
    args = parser_args()
    lr = args.lr
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    batch_size = args.batch_size
    epochs = args.epochs
    train_path = args.train_path
    test_path = args.test_path
    drop_out = args.drop_out
    score_function = OurSEScore(drop_out=drop_out)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(score_function.parameters(), lr=lr, betas=(beta_1,beta_2))
    train_dataset = load_dataset(train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tokenize_and_pad)
    test_dataset = load_dataset(test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=tokenize_and_pad)
    for epoch in range(epochs):
        score_function.train()
        train_losses = []
        for batch in train_dataloader:
            references_input_ids, references_attention_masks, predictions_input_ids, predictions_attention_masks, scores = batch
            predicted_scores = score_function.forward(references_input_ids, references_attention_masks,
                                                      predictions_input_ids, predictions_attention_masks)['score'].squeeze()
            loss = loss_function(predicted_scores, scores)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.detach().item())
        score_function.eval()
        with torch.no_grad():
            test_predictions = []
            test_real_scores = []
            for batch in test_dataloader:
                references_input_ids, references_attention_masks, predictions_input_ids, predictions_attention_masks, scores = batch
                predicted_scores = score_function.forward(references_input_ids, references_attention_masks,
                                                          predictions_input_ids, predictions_attention_masks)['score'].squeeze()
                test_predictions.append(predicted_scores)
                test_real_scores.append(scores)
            test_predictions = torch.stack(test_predictions, dim=0)
            test_real_scores = torch.stack(test_real_scores, dim=0)
            test_mse = loss_function(test_predictions, test_real_scores)
        print(f"For epoch {epoch} the average train loss {np.mean(train_losses)} | the test loss {test_mse}")
