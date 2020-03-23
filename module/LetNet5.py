import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from .layer1 import crxb_Conv2d
from .layer1 import crxb_Linear

# Todo: modify the Letnet structure
class LetNet5(nn.Module):
    def __init__(self, crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise,
                 enable_SAF, enable_ec_SAF):
        super(LetNet5, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)
        self.conv1 = crxb_Conv2d(1, 6, kernel_size=5, crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.conv2 = crxb_Conv2d(6, 16, kernel_size=5, crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.conv3 = crxb_Conv2d(16, 120, kernel_size=5, crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = crxb_Linear(120, 84, crxb_size=crxb_size, scaler_dw=scaler_dw,
                               gmax=gmax, gmin=gmin, gwire=gwire, gload=gload, freq=freq, temp=temp,
                               vdd=vdd, ir_drop=ir_drop, device=device, enable_noise=enable_noise,
                               enable_ec_SAF=enable_ec_SAF, enable_SAF=enable_SAF)
        self.fc2 = crxb_Linear(84, 10, crxb_size=crxb_size, scaler_dw=scaler_dw,
                               gmax=gmax, gmin=gmin, gwire=gwire, gload=gload, freq=freq, temp=temp,
                               vdd=vdd, ir_drop=ir_drop, device=device, enable_noise=enable_noise,
                               enable_ec_SAF=enable_ec_SAF, enable_SAF=enable_SAF)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
