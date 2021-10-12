import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from pathlib import Path
import os
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def main():
    BASE_DIR = Path(__file__).resolve().parent
    device = torch.device("cpu")
    path = os.path.join(BASE_DIR, 'model.pth')
    model = torch.load(path, map_location=device)
    MAX_LEN = 256
    # sad
    sent = 'i didnt feel humiliated'
    # sad
    sent = 'i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake'
    # anger
    sent = 'i am feeling grouchy'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = torch.tensor(
        [tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True)])
    attention_masks = []
    ## Create a mask of 1 for all input tokens and 0 for all padding tokens
    attention_masks = torch.tensor([[float(i > 0) for i in seq] for seq in input_ids])
    logits = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
    y_hat = F.softmax(logits[0][0], dim=0)

    result = np.round(y_hat.detach().numpy(), 5) * 100
    print(result)
if __name__ == '__main__':
    main()