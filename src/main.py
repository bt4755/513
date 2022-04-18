from __future__ import print_function
import argparse
import torch
from torch import cuda
from torch._C import NoneType
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbnn as bnn
import os
import numpy as np
from torchvision.transforms.transforms import Resize
from DCGAN import Discriminator, Generator, initialize_weights
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import Dataset_1.pokemon.pokemon.pokeset as pokemonDataset

torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.empty_cache()



if __name__=='__main__':
  torch.cuda.memory_summary(device=None, abbreviated=False)
  torch.cuda.empty_cache()
  D_SET = 'MNIST'
  EPOCHS = 10
  BATCH_SIZE = 5
  IMG_SIZE = 0
  if D_SET == 'poke':
      IMG_SIZE = 256
  else:
      IMG_SIZE = 64
  cuda = True and torch.cuda.is_available()
  Z = 100
  IMG_CHANNELS = 0
  if D_SET == 'poke':
      IMG_CHANNELS = 4
  else:
      IMG_CHANNELS = 1
  LEARNING_RATE = 3e-4
  device = None
  DATA_PATH = "./Dataset_1/pokemon/pokemon/"
  
  if cuda:
    torch.cuda.manual_seed(1)
  
  transforms = transforms.Compose(
    [
      transforms.Resize(IMG_SIZE),
      transforms.ToTensor(),
      transforms.Normalize(
        [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]
      )
    ]
  )

  #poke_set = pokemonDataset.pokemonDataset(DATA_PATH,721, transforms)
  trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transforms)
  data_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
  #data_loader = data.DataLoader(poke_set, batch_size=BATCH_SIZE)
  
  gen = Generator(Z, IMG_CHANNELS, IMG_SIZE, D_SET).cuda()
  disc = Discriminator( IMG_CHANNELS, IMG_SIZE, D_SET).cuda()
  initialize_weights(gen)
  initialize_weights(disc)
  torch.cuda.empty_cache()

  opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE,betas=(0.5, 0.999))
  opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE,betas=(0.5, 0.999))

  criterion = nn.BCELoss()

  fixed_noise = torch.randn((BATCH_SIZE, Z, 1,1)).cuda()
  writer_real = SummaryWriter(f"logs/real")
  writer_fake = SummaryWriter(f"logs/fake")
  step = 0


  torch.cuda.memory_summary(device=None, abbreviated=False)
  torch.cuda.empty_cache()
  gen.train()
  disc.train()
  for epoch in range(EPOCHS):
    for idx, (real,_) in enumerate(data_loader):
      #real = np.swapaxes(real, 1, 2)
      #real = np.swapaxes(real, 1, -1)
      #xwreal = real.to(torch.float)
      noise = torch.randn((BATCH_SIZE, Z, 1,1)).cuda()
      real = real.cuda()
      fake = gen(noise)
      #TRAIN DISCRIMINATOR
      disc_real = disc(real).reshape(-1)
      loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
      disc_fake = disc(fake).reshape(-1)
      loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
      loss_disc = (loss_disc_real + loss_disc_fake)/2
      disc.zero_grad()
      loss_disc.backward(retain_graph=True)
      opt_disc.step()

      #TRAIN GENERATOR
      output = disc(fake).reshape(-1)
      loss_gen = criterion(output, torch.ones_like(output))
      gen.zero_grad()
      loss_gen.backward()
      opt_gen.step()


      if idx % 100 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Batch [{idx}/{len(data_loader)}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")
        with torch.no_grad():
          fake = gen(fixed_noise)
          img_grd_real = utils.make_grid(real[:4], normalize=True)
          img_grd_fake = utils.make_grid(fake[:4], normalize=True)
          writer_real.add_image('Real', img_grd_real,global_step=step)
          writer_fake.add_image('Fake', img_grd_fake,global_step=step)
      step += 1


  


  

  
  