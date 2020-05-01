import os

import torch
import visdom
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

vis = visdom.Visdom(env="DAE_ReyhaneAskari")

if not os.path.exists('./DAEmlp_img'):
    os.mkdir('./DAEmlp_img')
if not os.path.exists('./DAEfilters'):
    os.mkdir('./DAEfilters')


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3


def add_noise(img):
    noise = torch.randn(img.size()) * 0.4
    noisy_img = img + noise
    return noisy_img


def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor: min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor: tensor_round(tensor))
])

dataset = MNIST('../../datasets', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        noisy_img = add_noise(img)
        noisy_img = Variable(noisy_img).cuda()
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(noisy_img)
        loss = criterion(output, img)
        MSE_loss = nn.MSELoss()(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.item(), MSE_loss.item()))
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        x_noisy = to_img(noisy_img.cpu().data)
        weights = to_img(model.encoder[0].weight.cpu().data)
        save_image(x, './DAEmlp_img/x_{}.png'.format(epoch))
        save_image(x_hat, './DAEmlp_img/x_hat_{}.png'.format(epoch))
        save_image(x_noisy, './DAEmlp_img/x_noisy_{}.png'.format(epoch))
        save_image(weights, './DAEfilters/epoch_{}.png'.format(epoch))
        vis.image(make_grid(x), win="x", opts=dict(title='x'))
        vis.image(make_grid(x_hat), "x_hat", opts=dict(title='x_hat'))
        vis.image(make_grid(x_noisy), "x_noisy", opts=dict(title='x_noisy'))
        vis.image(make_grid(weights), "weights", opts=dict(title='weights'))

torch.save(model.state_dict(), './sim_dautoencoder.pth')
