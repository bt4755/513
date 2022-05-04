import torch
import torch.nn as nn
import torchbnn as bnn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from skimage import io

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
      

