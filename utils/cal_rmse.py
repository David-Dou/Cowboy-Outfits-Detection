import torch
from torch import nn

class rmse:
    def __init__(self, features, model, labels):
        self.features = features
        self.model = model
        self.labels = labels

    def rmse_loss(self):
        mse_loss = nn.MSELoss()
        clipped_predictions = torch.clamp(self.model(self.features), 1, float("inf"))
        rmse_loss = torch.sqrt(mse_loss(torch.log(clipped_predictions), torch.log(self.labels)))
        '''
        tensor.item() to get its value, otherwise 
        '''
        return rmse_loss.item()