import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.ticker as ticker

# download data
batch_size = 128
image_size = 64

dataset = dset.CIFAR10(root='../../data/', download=True, train=True,
                       transform=transforms.Compose([transforms.Resize(image_size),
                                                    transforms.ToTensor()]))
# check device is cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model
class ConvVAE(nn.Module):
  '''
    nc: number of channels
    ngf: number of generator features
    ndf: number of discriminator features
    z: latent vector size
  '''
  def __init__(self, nc, ngf, ndf, z):
    
    super(ConvVAE, self).__init__()
    
    self.nc = nc
    self.ngf = ngf
    self.ndf = ndf
    self.z = z
    
    self.mean = nn.Linear(self.ndf*7*7, z)
    self.logvar = nn.Linear(self.ndf*7*7, z)
    self.d = nn.Linear(z, self.ngf*2*4*4)
    
    self.encoder = nn.Sequential(
        nn.Conv2d(nc, ndf*8, 4, 2, 0),
        nn.BatchNorm2d(ndf*8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf*8, ndf*4, 4, 2, 1),
        nn.BatchNorm2d(ndf*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf*4, ndf, 4, 2, 1),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(0.2, inplace=True)
    )
    
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(ngf*2, ngf*8, 4, 2, 1),
        nn.BatchNorm2d(ngf*8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(ngf*8, ngf*16, 4, 2, 1),
        nn.BatchNorm2d(ngf*16),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(ngf*16, ngf*32, 4, 2, 1),
        nn.BatchNorm2d(ngf*32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(ngf*32, nc, 4, 2, 1),
        nn.Sigmoid()
    )
    
  def encode(self, x):
    out = self.encoder(x)
    return self.mean(out.view(-1,self.ndf*7*7)), self.logvar(out.view(-1,self.ndf*7*7))
  
  def reparameterize(self, mu, logvar):
    if self.training:
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      return mu
  
  def decode(self, z):
    z = self.d(z)
    output = self.decoder(z.view(-1, self.ngf*2, 4, 4))
    return output
    
  def forward(self, input):
    mean, logvar = self.encode(input)
    z = self.reparameterize(mean, logvar)
    output = self.decode(z)
    return output, mean, logvar
    
# define loss function
def criterion(recon_x, x, mu, logvar):
  loss_func = nn.BCELoss(size_average=True)
  BCE = loss_func(recon_x, x)
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

  return BCE + KLD

# define train
def train(model, optimiser, epochs):
  losses = []
  for epoch in range(epochs):
    for idx, (data, label) in enumerate(dataloader):
      model.zero_grad()
      x = data.to(device)
      output, mean, logvar = model(x)
      loss = criterion(output, x, mean, logvar)
      losses.append(loss)
      loss.backward()
      optimiser.step()
      print('Done: [%d/%d][%d/%d] Loss: %.4f ' % (epoch, epochs, idx, len(dataloader), loss.item()))
  return losses

# initialize
cvae = ConvVAE(3, 64, 64, 10).to(device)
optimizer = torch.optim.Adam(cvae.parameters(), lr = 1e-3)

# train
losses = train(cvae, optimizer, 15)

# plot losses
plt.figure()
plt.plot(losses)
