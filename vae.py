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

dataset = dset.MNIST(root='../../data/', download=True, train=True,
                       transform=transforms.Compose([transforms.Resize(image_size),
                                                    transforms.ToTensor()]))
# check device is cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# show image
def show(img):
    npimg = img.numpy()
    ax = plt.gca()
    ax.grid(False)
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.axis('off')

# view training data
img = dataset.train_data[9]
show(make_grid([imglist], range=(0,1))) 

# define variational autoencoder
class VAE(nn.Module):
  
  def __init__(self):
    super(VAE, self).__init__()
    
    self.mean = nn.Linear(400, 20)
    self.logvar = nn.Linear(400, 20)
    
    self.encoder = nn.Sequential(
        nn.Linear(i**2, 1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(1024, 400)
    )
    
    self.decoder = nn.Sequential(
        nn.Linear(20, 64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(64, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(1024, i**2),
        nn.Sigmoid()
    )
  
  def encode(self, x):
    out = self.encoder(x)
    return self.mean(out), self.logvar(out)
  
  def reparameterize(self, mu, logvar):
    if self.training:
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      return mu
  
  def decode(self, z):
    output = self.decoder(z)
    return output
    
  def forward(self, input):
    mean, logvar = self.encode(input)
    z = self.reparameterize(mean, logvar)
    output = self.decode(z)
    return output, mean, logvar
  
# define loss function
def criterion(recon_x, x, mu, logvar):
  BCE = F.binary_cross_entropy(recon_x, x.view(-1, x.size(0)), size_average=True)
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return BCE + KLD

# define training method
def train(model, dataloader, optimiser, epochs):
  losses = []
  for epoch in range(epochs):
    for idx, (data, label) in enumerate(dataloader):
      model.zero_grad()
      x = data.view(data.size(0),-1).float().to(device)
      output, mu, logvar = model(x)
      loss = criterion(output, x, mu, logvar)
      losses.append(loss)
      loss.backward()
      optimiser.step()
      print('Done: [%d/%d][%d/%d] Loss: %.4f ' % (epoch, epochs, idx, len(dataloader), loss.item()))
  return losses

# initialise
vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr = 1e-3)

# train
losses = train(vae, dataloader, optimizer, 15)

# plot losses
plt.figure()
plt.plot(losses)
