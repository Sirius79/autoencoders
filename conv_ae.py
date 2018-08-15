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

# show image
def show(img):
  npimg = img.numpy()
  ax = plt.gca()
  ax.grid(False)
  plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
  plt.axis('off')

# define model
class ConvAutoEncoder(nn.Module):
  
  def __init__(self):
    
    super(ConvAutoEncoder, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 512, 4, 2, 0, bias=False), 
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(512, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 32, 4, 2, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(32, 8, 4, 2, 1, bias=False),
        nn.Sigmoid()
     )
    
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(8, 32, 4, 2, 0, bias=False), 
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(32, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(128, 512, 4, 2, 1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(512, 3, 4, 2, 1, bias=False),
        nn.Sigmoid()
     )
    
  def forward(self, input):
    encoded = self.encoder(input)
    output = self.decoder(encoded)
    return output
  
# define training method
def train(model, optimiser, criterion, epochs):
  losses = []
  for epoch in range(epochs):
    for idx, (data, label) in enumerate(dataloader):
      model.zero_grad()
      x = data.to(device)
      output = model(x)
      loss = criterion(output, x)
      losses.append(loss)
      loss.backward()
      optimiser.step()
      print('Done: [%d/%d][%d/%d] Loss: %.4f ' % (epoch, epochs, idx, len(dataloader), loss.item()))
  return losses

# define autoencoder
cae = ConvAutoEncoder().to(device)

# define optim and criterion
optimizer = torch.optim.Adam(cae.parameters(), lr = 0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

# train 
losses = train(cae, optimizer, criterion, 15)

# plot losses
plt.figure()
plt.plot(losses)
  
