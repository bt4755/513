from PIL import Image
from collections import OrderedDict, defaultdict
import tensorflow as tf
from tensorflow import Variable
import torchbnn as bnn
import bayesian_torch as bt
from bayesian_torch import layers as btl
import PIL
import os
from torch import cuda
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torchvision
import torch.optim as optim
from torch._C import NoneType
import torchbnn as bnn
import sys
import numpy as np
import timeit
import matplotlib.pyplot as plt
import matplotlib
import math
from torchvision.transforms.transforms import Resize, Compose, ToPILImage, ToTensor, Normalize
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset
from skimage import io
import torch.nn.init as init
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import glob

def save_state(model_gen, model_disc,  name):
    print('==> Saving model ...')
    state = {
              'state_dict_gen': model_gen.state_dict(),
              'state_dict_disc': model_disc.state_dict(),
            }
    for key in state['state_dict_gen'].keys():
        if 'module' in key:
            state['state_dict_gen'][key.replace('module.', '')] = \
                    state['state_dict_gen'].pop(key)
    for key in state['state_dict_disc'].keys():
        if 'module' in key:
            state['state_dict_disc'][key.replace('module.', '')] = \
                    state['state_dict_disc'].pop(key)
    torch.save(state, 'savedmodels/'+ name +'.best.pth.tar')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class pokemonDataset(Dataset):
  def __init__(self, root_dir, size, transform):
      self.root_dir = root_dir
      self.transform = transform
      self.size = size

  def __getitem__(self, index):
      img = io.imread("{root}/{idx}.png".format(root= self.root_dir, idx = index + 1))
      img = self.transform(img)
      return img, 0

  def __len__(self):
      return self.size

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            #nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
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

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class Bayes_Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Bayes_Discriminator, self).__init__()
        self.cdim = channels_img
        self.gfdim = features_d
        prior_mu = 0.0
        prior_sigma = 1.0
        posterior_mu_init = 0.0
        posterior_rho_init = -3.0
        # input: N x channels_img x 64 x 64
        self.conv1 = btl.Conv2dReparameterization(in_channels=channels_img, 
                                                  out_channels=features_d, 
                                                  kernel_size=4, stride=2, 
                                                  padding=1, prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init, bias=False)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = btl.Conv2dReparameterization(in_channels=features_d, 
                                                  out_channels=features_d * 2, 
                                                  kernel_size=4, stride=2, 
                                                  padding=1, prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=features_d * 2)
        self.relu2 = nn.LeakyReLU(0.2)

        self.conv3 = btl.Conv2dReparameterization(in_channels=features_d*2, 
                                                  out_channels=features_d * 4, 
                                                  kernel_size=4, stride=2, 
                                                  padding=1, prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=features_d * 4)
        self.relu3 = nn.LeakyReLU(0.2)

        self.conv4 = btl.Conv2dReparameterization(in_channels=features_d * 4, 
                                                  out_channels=features_d * 8, 
                                                  kernel_size=4, stride=2, 
                                                  padding=1, prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=features_d * 8)
        self.relu4 = nn.LeakyReLU(0.2)
        # After all _block img output is 4x4 (Conv2d below makes into 1x1)
        self.conv5 = btl.Conv2dReparameterization(in_channels=features_d*8, 
                                                  out_channels=1, kernel_size=4, 
                                                  stride=2, padding=0, prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init, bias=False)

        #nn.Sigmoid(),

    def forward(self, x):
      #print(x.shape)
      kl_sum = 0
      out, kl = self.conv1(x)
      kl_sum += kl
      out = self.relu1(out)

      out, kl = self.conv2(out)
      kl_sum += kl
      out = self.bn1(out)
      out = self.relu2(out)

      out, kl = self.conv3(out)
      kl_sum += kl
      out = self.bn2(out)
      out = self.relu3(out)

      out, kl = self.conv4(out)
      kl_sum += kl
      out = self.bn3(out)
      out = self.relu4(out)

      out, kl = self.conv5(out)
      kl_sum += kl
      return out, kl_sum


class Bayes_Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Bayes_Generator, self).__init__()
        self.zdim= channels_noise
        self.cdim = channels_img
        self.gfdim = features_g
        prior_mu = 0.0
        prior_sigma = 1.0
        posterior_mu_init = 0.0
        posterior_rho_init = -3.0
        # Input: N x channels_noise x 1 x 1
        self.convT1 = btl.ConvTranspose2dReparameterization(in_channels=channels_noise,
                                                            out_channels=features_g * 16,kernel_size=4,
                                                            stride=1,padding=0, prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,bias=False)
        self.bn1 = nn.BatchNorm2d(features_g * 16)

        self.convT2 = btl.ConvTranspose2dReparameterization(in_channels=features_g * 16,
                                                            out_channels=features_g * 8,kernel_size=4,
                                                            stride=2,padding=1, prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,bias=False)
        self.bn2 = nn.BatchNorm2d(features_g * 8)

        self.convT3 = btl.ConvTranspose2dReparameterization(in_channels=features_g * 8,
                                                            out_channels=features_g * 4,kernel_size=4,
                                                            stride=2,padding=1, prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,bias=False)
        self.bn3 = nn.BatchNorm2d(features_g * 4)

        self.convT4 = btl.ConvTranspose2dReparameterization(in_channels=features_g * 4,
                                                            out_channels=features_g * 2,kernel_size=4,
                                                            stride=2,padding=1, prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,bias=False)
        self.bn4 = nn.BatchNorm2d(features_g * 2)

        self.convT5 = btl.ConvTranspose2dReparameterization(in_channels=features_g * 2,
                                                            out_channels=channels_img,kernel_size=4,
                                                            stride=2,padding=1, prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,bias=False)

    def forward(self, x):
        kl_sum = 0

        #print(x.shape)
        out, kl = self.convT1(x)
        kl_sum += kl
        out = self.bn1(out)
        out = F.relu(out)

        #print(out.shape)
        out, kl = self.convT2(out)
        kl_sum += kl
        out = self.bn2(out)
        out = F.relu(out)

        #print(out.shape)
        out, kl = self.convT3(out)
        kl_sum += kl
        out = self.bn3(out)
        out = F.relu(out)

        #print(out.shape)
        out, kl = self.convT4(out)
        kl_sum += kl
        out = self.bn4(out)
        out = F.relu(out)

        #print(out.shape)
        out, kl = self.convT5(out)
        kl_sum += kl
        out = F.tanh(out)

        #print(out.shape)
        return out, kl_sum

if __name__=='__main__':
  # Hyperparameters etc.
  Bayes = False
  epochs = 100
  img_size = 64
  noise_dim = 100
  feature_size = 16
  dset = 'MNIST'
  #dset ='POKE'

  batch_size = 128
  channels = 0
  if dset == 'MNIST':
    channels = 1
  else :
    channels = 4

  lr = 2e-4
  path = None
  if dset == 'MNIST':
    path = "Datasets"
  else :
    path = "Datasets/pokemon/pokemon/"


    
  torch.cuda.manual_seed(1)

  transformer = None
  dataset = None
  if dset == 'MNIST':
    transformer = Compose(
      [
        Resize(img_size),
        ToTensor(),
        Normalize(
          [0.5 for _ in range(channels)], [0.5 for _ in range(channels)]
        )
      ]
    )

    dataset = datasets.MNIST(root=path, train=True, download=True, transform=transformer)
  else:
    transformer = Compose(
      [
        ToPILImage(),
        Resize(img_size),
        ToTensor(),
        Normalize(
          [0.5 for _ in range(channels)], [0.5 for _ in range(channels)]
        )
      ]
    )
    dataset = pokemonDataset(path,721, transformer)

  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  gen = None
  disc = None
  if Bayes:
    gen = Bayes_Generator(noise_dim, channels, feature_size).to(device)
    disc = Bayes_Discriminator(channels, feature_size).to(device)
  else:
    gen = Generator(noise_dim, channels, feature_size).to(device)
    disc = Discriminator(channels, feature_size).to(device)
    initialize_weights(gen)
    initialize_weights(disc)
  

  opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
  opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
  criterion = nn.BCEWithLogitsLoss()

  fixed_noise = torch.randn((batch_size, noise_dim, 1,1)).to(device)
  writer_real = SummaryWriter('logs/{dset}/epochs{ep}_noisedim{dim}_img_size{size}_feature_size{fsize}/Real'.format(dset=dset,ep=epochs, dim=noise_dim, size=img_size, fsize=feature_size ))
  writer_fake = SummaryWriter('logs/{dset}/epochs{ep}_noisedim{dim}_img_size{size}_feature_size{fsize}/Fake'.format(dset=dset,ep=epochs, dim=noise_dim, size=img_size, fsize=feature_size ))
  step = 0

  gen.train()
  disc.train()

  if Bayes:
    for epoch in range(epochs):
      for batch_idx, (real, _) in enumerate(dataloader):
          real = real.to(device)
          noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
          fake , kl_gen = gen(noise)
          out, kl_disc = disc(real)
          disc_real = out.reshape(-1)
          loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
          out2, kl = disc(fake)
          disc_fake = out2.reshape(-1)
          loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
          loss_disc = (loss_disc_real + loss_disc_fake) / 2
          disc.zero_grad()
          loss_disc.backward(retain_graph=True)
          opt_disc.step()
          out3 , kls = disc(fake)
          output = out3.reshape(-1)
          loss_gen = criterion(output, torch.ones_like(output))
          gen.zero_grad()
          loss_gen.backward()
          opt_gen.step()


          if batch_idx % 100 == 0:
              print(
                  f"Epoch [{epoch + 1}/{epochs}] Batch {batch_idx}/{len(dataloader)} \
                    Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
              )

              with torch.no_grad():
                  fake, lk = gen(fixed_noise)
                  img_grid_real = torchvision.utils.make_grid(
                      real[:32], normalize=True
                  )
                  img_grid_fake = torchvision.utils.make_grid(
                      fake[:32], normalize=True
                  )

                  writer_real.add_image("Real", img_grid_real, global_step=step)
                  writer_fake.add_image("Fake", img_grid_fake, global_step=step)

              step += 1
    name = "BayesModel_{dset}_epochs_{ep}_img_size_{size}_feature_size_{fsize}".format(dset=dset,ep=epochs, size=img_size, fsize=feature_size )
    save_state(gen, disc, name)

  else:
    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()


            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1
    name = "Model_{dset}_epochs_{ep}_img_size_{size}_feature_size_{fsize}".format(dset=dset,ep=epochs, size=img_size, fsize=feature_size )
    save_state(gen, disc, name)


  


  

  
  