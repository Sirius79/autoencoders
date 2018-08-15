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

# define simple autoencoder
class AutoEncoder(nn.Module):
  
  def __init__(self):
    super(AutoEncoder, self).__init__()
    
    self.encoder = nn.Sequential(
        nn.Linear(image_size**2, 1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(1024, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(64, 12),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(12, 3),
        nn.Sigmoid()
    )
    
    self.decoder = nn.Sequential(
        nn.Linear(3, 12),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(12, 64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(64, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(1024, image_size**2),
        nn.Sigmoid()
    )
  
  def forward(self, input):
    output = self.decoder(self.encoder(input))
    return output

# define training method
def train(model, optimiser, criterion, epochs):
  losses = []
  for epoch in range(epochs):
    for idx, (data, label) in enumerate(dataloader):
      model.zero_grad()
      x = data.view(data.size(0),-1).float().to(device)
      output = model(x)
      loss = criterion(output, x)
      losses.append(loss)
      loss.backward()
      optimiser.step()
      print('Done: [%d/%d][%d/%d] Loss: %.4f ' % (epoch, epochs, idx, len(dataloader), loss.item()))
  return losses

# define autoencoder
ae = AutoEncoder().to(device)

# define optim and criterion
optimizer = torch.optim.Adam(ae.parameters(), lr = 0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

# train 
losses = train(ae, optimizer, criterion, 15)

# plot losses
plt.figure()
plt.plot(losses)
  
