import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

class DebertaPureRegressor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path)
        self.deberta = DebertaV2Model.from_pretrained(model_path, config=self.config)
        self.regression_head = nn.Sequential(
            # nn.Linear(self.config.hidden_size, 256), nn.ReLU(), nn.Dropout(0.1),
            # nn.Linear(256, 64), nn.ReLU(), 
            # nn.Linear(64, 3),
            # nn.Tanh()
            
            nn.Linear(self.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 3)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regression_head(cls_output)

class LlamaEmotionalProjector(nn.Module):
    def __init__(self, llama_dim=3072):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(3, 512), nn.GELU(),
            nn.Linear(512, llama_dim), 
            nn.LayerNorm(llama_dim)
        )
    def forward(self, pad_vector):
        return self.projector(pad_vector).unsqueeze(1)