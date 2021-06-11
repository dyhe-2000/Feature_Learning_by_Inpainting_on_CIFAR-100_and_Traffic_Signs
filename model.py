import torch
import torch.nn as nn
from torch.nn.modules.normalization import GroupNorm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
import torch.nn.functional as F  # useful stateless functions

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (4,4), stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, (4,4), stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, (4,4), stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, (4,4), stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, (4,4), stride=2, padding=1)
        self.conv6 = nn.Conv2d(512, 4000, (4,4), stride=2, padding=0)
        self.trconv1 = nn.ConvTranspose2d(4000, 512, (4,4), stride=2, padding=0)
        self.trconv2 = nn.ConvTranspose2d(512, 256, (4,4), stride=2, padding=1)
        self.trconv3 = nn.ConvTranspose2d(256, 128, (4,4), stride=2, padding=1)
        self.trconv4 = nn.ConvTranspose2d(128, 64, (4,4), stride=2, padding=1)
        self.trconv5 = nn.ConvTranspose2d(64, 3, (4,4), stride=2, padding=1)

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.trconv1(x)
        x = F.relu(x)
        x = self.trconv2(x)
        x = F.relu(x)
        x = self.trconv3(x)
        x = F.relu(x)
        x = self.trconv4(x)
        x = F.relu(x)
        x = self.trconv5(x)
        return x
        
def test_EncoderDecoder():
    the_encoder_decoder = EncoderDecoder()
    test_input = torch.rand(1, 3, 128, 128)
    output = the_encoder_decoder.forward(test_input)
    assert output.size() == (1,3,64,64)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (4,4), stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, (4,4), stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, (4,4), stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, (4,4), stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, (4,4), stride=2, padding=0)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.sig(x)
        return x
        
def test_Discriminator():
    the_discriminator = Discriminator()
    test_input = torch.rand(1, 3, 64, 64)
    output = the_discriminator.forward(test_input)
    assert output.size() == (1,1,1,1)
    
# test_EncoderDecoder()
# test_Discriminator()