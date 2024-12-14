# %%
import torch
from torch import nn
import torch.nn.functional as F
# %% A network that models the output as a Gaussian latent variable
class BayesNet(nn.Module):
    def __init__(self):
        super().__init__()
        
# %%