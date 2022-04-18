import torch
import torch
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
import torchbnn as bnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

 
class Discriminator(nn.Module):
  def __init__(self, in_channels, out_channels, dset):
      super(Discriminator, self).__init__()
      if dset == 'poke':
        self.discriminator = nn.Sequential(
        # Pokemon 819 * in_channels * 256 * 256
        nn.Conv2d( in_channels, out_channels,4, 2, 1),
        nn.LeakyReLU(0.2),
        # Pokemon 819 * out_channels * 128 * 128
        nn.Conv2d( out_channels, out_channels*2, 4, 2, 1),
        nn.BatchNorm2d(out_channels*2),
        nn.LeakyReLU(0.2),
        # Pokemon 819 * out_channels*2 * 64 * 64 
        nn.Conv2d( out_channels*2, out_channels*4, 4, 2, 1),
        nn.BatchNorm2d(out_channels*4),
        nn.LeakyReLU(0.2),
        # Pokemon 819 * out_channels*4 * 32 * 32
        nn.Conv2d( out_channels*4, out_channels*8, 4, 2, 1),
        nn.BatchNorm2d(out_channels*8),
        nn.LeakyReLU(0.2),
        # Pok   emon 819 * out_channels*8 * 16 * 16 
        nn.Conv2d( out_channels*8, out_channels*16, 4, 2, 1),
        nn.BatchNorm2d(out_channels*16),
        nn.LeakyReLU(0.2),
        # Pokemon 819 * out_channels*16 * 8 * 8 
        nn.Conv2d( out_channels*16, out_channels*32, 4, 2, 1),
        nn.BatchNorm2d(out_channels*32),
        nn.LeakyReLU(0.2),
        # Pokemon 819 * out_channels*32 * 4 * 4
        nn.Conv2d( out_channels*32, 1, 4, 2, 0),
        nn.Sigmoid()
        # To ensure single value between 1 and 0
        )
      else:
        self.discriminator = nn.Sequential(
        # Pokemon 819 * out_channels*2 * 64 * 64 
        nn.Conv2d( in_channels, out_channels,4, 2, 1),
        nn.LeakyReLU(0.2),
        # Pokemon 819 * out_channels*4 * 32 * 32
        nn.Conv2d(out_channels, out_channels*2, 4, 2, 1),
        nn.BatchNorm2d(out_channels*2),
        nn.LeakyReLU(0.2),
        # Pok   emon 819 * out_channels*8 * 16 * 16 
        nn.Conv2d( out_channels*2, out_channels*4, 4, 2, 1),
        nn.BatchNorm2d(out_channels*4),
        nn.LeakyReLU(0.2),
        # Pokemon 819 * out_channels*16 * 8 * 8 
        nn.Conv2d(out_channels*4, out_channels*8, 4, 2, 1),
        nn.BatchNorm2d(out_channels*8),
        nn.LeakyReLU(0.2),
        # Pokemon 819 * out_channels*32 * 4 * 4
        nn.Conv2d( out_channels*8, 1, 4, 2, 0),
        nn.Sigmoid()
        )
        


  # To run input through forward pass of nn run this
  def forward(self, x):
      return self.discriminator(x)


class Generator(nn.Module):
  def __init__(self, z_dim, in_channels, out_channels, dset) -> None:
      super(Generator, self).__init__()
      if dset == 'poke':
        self.gen = nn.Sequential(
          # Input: N x in_channels x 1 x 1
          self.convBlockBatch(z_dim, out_channels*64, 4, 1,0),
          # N x out_channels x 4 x 4
          self.convBlockBatch(out_channels*64, out_channels*32,  4,2,1),
          # N x out_channels x 8 x 8
          self.convBlockBatch(out_channels*32, out_channels*16,  4,2,1),
          # N x out_channels x 16 x 16
          self.convBlockBatch(out_channels*16, out_channels*8,  4,2,1),
          # N x out_channels x 32 x 32
          self.convBlockBatch(out_channels*8, out_channels*4,  4,2,1),
          # N x out_channels x 64 x 64
          self.convBlockBatch(out_channels*4, out_channels*2,  4,2,1),
          # N x out_channels x 128 x 128
          nn.ConvTranspose2d(out_channels*2, in_channels,  4,2,1),
          nn.Tanh()
          # N x out_channels x 256 x 256
        )
      else:
        self.gen = nn.Sequential(
          # Input: N x in_channels x 1 x 1
          self.convBlockBatch(z_dim, out_channels*16, 4, 1,0),
          self.convBlockBatch(out_channels*16, out_channels*8,  4,2,1),
          # N x out_channels x 32 x 32
          self.convBlockBatch(out_channels*8, out_channels*4,  4,2,1),
          # N x out_channels x 64 x 64
          self.convBlockBatch(out_channels*4, out_channels*2,  4,2,1),
          # N x out_channels x 128 x 128
          nn.ConvTranspose2d(out_channels*2, in_channels,  4,2,1),
          nn.Tanh()
          # N x out_channels x 256 x 256
        )

  def forward(self, x):
    return self.gen(x)

  def convBlockBatch(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
      nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=False,
      ),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
    )


def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, (Conv2d, ConvTranspose2d, BatchNorm2d)):
      nn.init.normal_(m.weight.data, 0.0, 0.02)
      

      
