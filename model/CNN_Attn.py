# # Model for CNN-Attn SLF Estimator
from torch import nn
from torch.nn import functional as F
import torch
from model.attention_conv import AugmentedConv


def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, activation='LeakyReLU', batch_norm=True):
    layer_out = []
    bias = True
    if batch_norm:
        bias = False
    layer_out.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    if batch_norm:
        layer_out.append(nn.BatchNorm2d(out_channels))
    if activation == 'LeakyReLU':
        layer_out.append(nn.LeakyReLU(0.2))
    elif activation == 'ReLU':
        layer_out.append(nn.ReLU())
    elif activation == 'Linear':
        pass
    elif activation == 'Sigmoid':
        layer_out.append(nn.Sigmoid())
    else:
        raise NotImplementedError('Activation Function {} not understood.'.format(activation))
    return nn.Sequential(*layer_out)


# CNN-attention estimator
class CNN_Attn(nn.Module):
    def __init__(self, M, P, K):
        super(CNN_Attn, self).__init__()
        self.M = M  # The number of RF nodes
        self.N = int(M*(M-1))  # The number of links
        self.P = P  # Each node has P measurement positions
        self.K = K  # dimension of SLF image
        # input size (CxWxH) = (NxPxP)
        # output size (CxWxH) = (1xK0xK1)
        self.conv1 = conv_layer(self.N, 64, 3, padding='same', activation='LeakyReLU')              # shape: [batch_size, 64, P, P]

        self.augmented_conv1 = AugmentedConv(in_channels=64, out_channels=64, kernel_size=3, dk=128, dv=8, Nh=4,
                                             shape=int(self.P),
                                             relative=True)  # shape: [batch_size, 64, P-2, P-2]
        self.bn1 = nn.BatchNorm2d(64)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.dropout1 = nn.Dropout2d(0.5)

        self.conv2 = conv_layer(64, 128, 3, padding='same', activation='LeakyReLU')                 # shape: [batch_size, 64, P, P]
        self.conv3 = conv_layer(128, 256, 3, padding='valid', activation='LeakyReLU')              # shape: [batch_size, 128, P-2, P-2]

        self.augmented_conv2 = AugmentedConv(in_channels=256, out_channels=256, kernel_size=3, dk=512, dv=64, Nh=4,
                                             shape=int(self.P-2),
                                             relative=True)  # shape: [batch_size, 256, P-2, P-2]
        self.bn2 = nn.BatchNorm2d(256)
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.dropout2 = nn.Dropout2d(0.5)

        self.conv4 = conv_layer(256, 512, 3, padding='same', activation='LeakyReLU')               # shape: [batch_size, 512, P-2, P-2]
        self.conv5 = conv_layer(512, 2048, 1, padding=0, activation='LeakyReLU')
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout1_linear = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(2048, self.K[0]*self.K[1])  # shape: [batch_size, K0xK1]

    def forward(self, x):
        out = self.conv1(x)                               # shape: [batch_size, 64, P, P]
        attn = self.augmented_conv1(out)                  # shape: [batch_size, 64, P, P]
        attn = F.leaky_relu(self.bn1(attn), 0.2)
        out = torch.sigmoid(self.gamma1) * self.dropout1(attn) + (1-torch.sigmoid(self.gamma1)) * out      # shape: [batch_size, 64, P, P]
        out = self.conv2(out)                             # shape: [batch_size, 128, P, P]
        out = self.conv3(out)                             # shape: [batch_size, 256, P-2, P-2]
        attn = self.augmented_conv2(out)                  # shape: [batch_size, 256, P-2, P-2]
        attn = F.leaky_relu(self.bn2(attn), 0.2)
        out = torch.sigmoid(self.gamma2) * self.dropout2(attn) + (1-torch.sigmoid(self.gamma2)) * out      # shape: [batch_size, 256, P-2, P-2]
        out = self.conv4(out)                             # shape: [batch_size, 512, 4, 4]
        out = self.conv5(out)                             # shape: [batch_size, 2048, 4, 4]
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dropout1_linear(out)
        out = self.linear1(out)
        out = torch.sigmoid(out)
        out = out.view(-1, 1, self.K[0], self.K[1])
        return out