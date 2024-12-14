# %%
import torch
from torch import nn
import torch.nn.functional as F
# %%
class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.linear2 = nn.Linear(width, width)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        h = self.bn1(h)
        h = self.linear2(h)
        return F.relu(h + x)

class ConcatNet(nn.Module):
    def __init__(self, width, lambda_wt=0., outputs=1):
        super().__init__()
        self.lambda_wt = lambda_wt
        self.eb_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9*9, width),
            nn.LeakyReLU(),
            ResBlock(width),
            nn.Linear(width, width)
        )
        self.hb_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*4, width),
            nn.LeakyReLU(),
            ResBlock(width),
            nn.Linear(width, width)
        )
        self.ho_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*3, width),
            nn.LeakyReLU(),
            ResBlock(width),
            nn.Linear(width, width)
        )

        self.net = nn.Sequential(
            nn.Linear(width*3, width*3),
            nn.LeakyReLU(),
            ResBlock(width*3),
            ResBlock(width*3),
            nn.Linear(width*3, outputs)
        )
    
    def forward(self, x_eb, x_hb, x_ho):
        z_eb = self.eb_net(x_eb)
        z_hb = self.hb_net(x_hb)
        z_ho = self.ho_net(x_ho)
        z = nn.functional.leaky_relu(torch.cat([z_eb, z_hb, z_ho], 1))

        return self.net(z)

class MeanRegularizedLoss(nn.Module):
    def __init__(self, batch_size, lambda_weight=1.0, device=torch.device("cuda:0")):
        super(MeanRegularizedLoss, self).__init__()
        self.batch_size = batch_size
        self.lambda_weight = lambda_weight
        conv_wt = torch.ones([1, 1, self.batch_size]).to(device)/self.batch_size
        self.register_buffer("conv_wt", conv_wt)
    
    def forward(self, predictions, targets):
        # Mean Squared Error
        mse_loss = torch.mean((predictions - targets) ** 2)
        
        mean_predictions = F.conv1d(predictions.view(1, 1, -1), self.conv_wt, 
                                    stride=self.batch_size)
        mean_targets = F.conv1d(targets.view(1, 1, -1), self.conv_wt, 
                                stride=self.batch_size)
        
        bias_penalty = torch.mean((mean_predictions - mean_targets) ** 2)
        
        # Combined Loss
        total_loss = mse_loss + self.lambda_weight * bias_penalty
        return total_loss    
