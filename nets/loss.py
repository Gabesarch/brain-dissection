import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

import ipdb
st = ipdb.set_trace

class OptRespLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    def forward(self, features, y_brain, mask, compute_corr=False):
        
        masked_features = features[mask]
        masked_y = y_brain[mask]

        loss = self.criterion(masked_features, masked_y)

        if compute_corr:
            corr, _ = pearsonr(masked_features.detach().cpu().numpy(), masked_y.detach().cpu().numpy())
            corr = torch.tensor(corr).to(masked_features.device)
            return loss, corr
        
        return loss

