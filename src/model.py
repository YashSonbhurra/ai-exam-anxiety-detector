import torch
import torch.nn as nn
from transformers import BertModel

class BertAnxietyClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.classifier(x)
