#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy as np
# from emoclassfer_2.EMOClassifer import emoClassifer
from pathlib import Path
import os

# bertmodel = BertModel.from_pretrained("beomi/kcbert-large")
# tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-large")
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.__version__)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
BASE_DIR = Path(__file__).resolve().parent
device = torch.device("cpu")
print(BASE_DIR)
path_kcbert = os.path.join(BASE_DIR, 'articleapp/kcbert_tokenizer.pth')
path_epoch20 = os.path.join(BASE_DIR, 'articleapp/epoch20.pth')
class emoClassifer(torch.nn.Module):
    def __init__(self, bert: BertModel, num_class: int, device: torch.device):
        super().__init__()
        self.bert = bert
        self.H = bert.config.hidden_size
        self.W_hy = torch.nn.Linear(self.H, num_class)  # (H, 3)
        self.to(device)

    def forward(self, X: torch.Tensor):
        """
        :param X:
        :return:
        """
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert(input_ids, token_type_ids, attention_mask)[0]

        return H_all

    def predict(self, X):
        H_all = self.forward(X)  # N, L, H
        H_cls = H_all[:, 0, :]  # 첫번째(cls)만 가져옴 (N,H)
        # N,H  H,3 -> N,3

        y_hat = self.W_hy(H_cls)
        return y_hat  # N,3

    def training_step(self, X, y):
        '''
        :param X:
        :param y:
        :return: loss
        '''
        # y = torch.LongTensor(y)
        y_pred = self.predict(X)
        y_pred = F.softmax(y_pred, dim=1)
        # loss
        loss = F.cross_entropy(y_pred, y).sum()
        return loss

def Build_X (sents, tokenizer, device):
    X = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    return torch.stack([
        X['input_ids'],
        X['token_type_ids'],
        X['attention_mask']
    ], dim=1).to(device)

def predict(DATA):
    device = torch.device('cpu')
    # bertmodel = BertModel.from_pretrained("beomi/kcbert-base")
    # tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
    # bertmodel = torch.load('kcbertmodel.pth', map_location=device)
    with open(path_kcbert,'rb') as f:

        tokenizer = torch.load(f, map_location=device)
        with open(path_epoch20,'rb') as f2:
            model = torch.load(f2, map_location=device)
            ############
            # tokenizer = torch.load(path_kcbert, map_location=device)
            #
            # model = torch.load(path_epoch20, map_location=device)
            # model.eval()
            X = Build_X(DATA, tokenizer, device)
            print(X)
            y_hat = model.predict(X)
            y_hat = F.softmax(y_hat, dim=1)

            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)
            result = list( map(float, list(y_hat.detach().numpy()[0] * 100)))
            return result

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gis_6ban_1.settings.local')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

