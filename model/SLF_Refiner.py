# Model for SLF_Refiner
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


# SLF-refiner
# overall structure
class SLF_Refiner(nn.Module):
    def __init__(self, M, P, K):
        super(SLF_Refiner, self).__init__()
        self.M = M  # The number of RF nodes
        self.P = P  # Each node has P measurement positions
        self.K = K  # dimension of SLF image

        self.encoder = Encoder(self.K)
        self.generator = Generator(self.M, self.P)
        self.decoder = Decoder(self.K)

        for m in self.modules():
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('Linear') != -1:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, rss, slf_coarse):
        encoded_slf = self.encoder(slf_coarse)
        gen_out = self.generator(rss)
        slf_recon = self.decoder(gen_out, encoded_slf)
        return slf_recon


# Encoder part
class Encoder(nn.Module):
    def __init__(self, K):
        super(Encoder, self).__init__()
        self.K = K  # dimension of SLF image
        # input size (CxWxH) = (1xK0xK1)
        # output size 1024, 1024, (512, K0/8-1, K1/8-1)=512x4x4

        # Encoding coarse slf image x
        self.conv1 = conv_layer(1, 16, 5, stride=2, padding=2)         # shape: [batch_size, 16, K0/2, K1/2]

        self.conv2 = nn.Conv2d(16, 32, 3, padding='same')             # shape: [batch_size, 32, K0/2, K1/2]
        self.bn1 = nn.BatchNorm2d(32)
        self.augmented_conv1 = AugmentedConv(in_channels=32, out_channels=32, kernel_size=3, dk=64, dv=8,
                                             Nh=2)  # shape: [batch_size, 32, K0/2, K1/2]
        # self.augmented_conv1 = nn.Conv2d(32, 32, 3, padding=1)  # shape: [batch_size, 32, K0/2, K1/2]
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = conv_layer(32, 64, 3, padding='same')            # shape: [batch_size, 64, K0/2, K1/2]
        self.maxpool1 = nn.MaxPool2d(2, 2)  # shape: [batch_size, 64, K0/4, K1/4]

        self.conv4 = nn.Conv2d(64, 128, 3, padding=0)                 # shape: [batch_size, 128, K0/4-2, K1/4-2]
        self.bn3 = nn.BatchNorm2d(128)
        self.augmented_conv2 = AugmentedConv(in_channels=128, out_channels=128, kernel_size=3, dk=256, dv=32,
                                             Nh=2)  # shape: [batch_size, 128, K0/4-2, K1/4-2]
        # self.augmented_conv2 = nn.Conv2d(128, 128, 3, padding=1)  # shape: [batch_size, 128, K0/4-2, K1/4-2]
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = conv_layer(128, 128, 3, padding='same')  # shape: [batch_size, 128, K0/4-2, K1/4-2]
        self.maxpool2 = nn.MaxPool2d(2, 2)                             # shape: [batch_size, 128, K0/8-1, K1/8-1]
        self.conv6 = conv_layer(128, 128, 3, padding='same')           # shape: [batch_size, 128, K0/8-1, K1/8-1]

    def forward(self, x):
        # x is the coarse slf image
        out = self.conv1(x)

        out = self.conv2(out)
        attn = self.augmented_conv1(F.leaky_relu(self.bn1(out), 0.2))
        out = torch.sigmoid(self.gamma1) * attn + (
                1 - torch.sigmoid(self.gamma1)) * out
        out = F.leaky_relu(self.bn2(out), 0.2)

        out = self.conv3(out)
        out = self.maxpool1(out)

        out = self.conv4(out)
        attn = self.augmented_conv2(F.leaky_relu(self.bn3(out), 0.2))
        out = torch.sigmoid(self.gamma2) * attn + (
                    1 - torch.sigmoid(self.gamma2)) * out
        out = F.leaky_relu(self.bn4(out), 0.2)

        out = self.conv5(out)
        out = self.maxpool2(out)
        out = self.conv6(out)
        return out


# Generator
class Generator(nn.Module):
    def __init__(self, M, P):
        super(Generator, self).__init__()
        self.M = M  # The number of RF nodes
        self.N = int(M * (M - 1))  # The number of links
        self.P = P  # Each node has P measurement positions
        # input size (CxWxH) = (NxPxP)
        self.conv1 = nn.Conv2d(self.N, 32, 3, padding='same')  # shape: [batch_size, 32, P, P]
        self.bn1 = nn.BatchNorm2d(32)

        self.augmented_conv1 = AugmentedConv(in_channels=32, out_channels=32, kernel_size=3, dk=64, dv=8,
                                             Nh=2)  # shape: [batch_size, 128, P, P]
        # self.augmented_conv1 = nn.Conv2d(32, 32, 3, padding=1)  # shape: [batch_size, 32, P, P]
        self.bn2 = nn.BatchNorm2d(32)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.conv2 = conv_layer(32, 64, 3, padding=0)  # shape: [batch_size, 64, P-2, P-2]
        self.conv3 = conv_layer(64, 128, 3, padding=1)  # shape: [batch_size, 128, P-2, P-2]

    def forward(self, x):
        # during testing stage
        # x is the RSS measurement
        out = self.conv1(x)  # shape: [batch_size, 32, P, P]
        attn = self.augmented_conv1(F.leaky_relu(self.bn1(out), 0.2))  # shape: [batch_size, 32, P, P]
        out = torch.sigmoid(self.gamma1) * attn + (
                1 - torch.sigmoid(self.gamma1)) * out  # shape: [batch_size, 32, P, P]
        out = F.leaky_relu(self.bn2(out), 0.2)
        out = self.conv2(out)  # shape: [batch_size, 64, P-2, P-2]
        out = self.conv3(out)  # shape: [batch_size, 128, P-2, P-2]
        return out


# Decoder
class Decoder(nn.Module):
    def __init__(self, K):
        super(Decoder, self).__init__()
        self.K = K  # dimension of SLF image
        # input: x, generator_x, encoder_x
        # size: [1, K0, K1], [batch_size, 128, P-2, P-2], [128, K0/8-1, K1/8-1]
        self.conv1 = conv_layer(128, 128, 3, padding=1)
        self.conv2 = conv_layer(256, 256, 3, padding=1)  # shape: [batch_size, 256, 4, 4]
        self.conv3 = conv_layer(256, 512, 3, padding=1)  # shape: [batch_size, 512, 4, 4]
        self.conv4 = conv_layer(512, 2048, 1, padding=0)         # shape: [batch_size, 2048, 4, 4]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout_linear = nn.Dropout(p=0.5)
        self.linear_out = nn.Linear(2048, self.K[0] * self.K[1])

    def forward(self, generator_x, encoder_x):
        # x: coarse slf image
        # generator_x: output of the generator
        # encoder_x: output of the encode_x()
        out = self.conv1(generator_x)                  # shape: [batch_size, 128, 4, 4]
        out = torch.cat((out, encoder_x), dim=1)     # shape: [batch_size, 128+128, 4, 4]
        out = self.conv2(out)      # shape: [batch_size, 256, 4, 4]
        out = self.conv3(out)  # shape: [batch_size, 512, 4, 4]
        out = self.conv4(out)      # shape: [batch_size, 2048, 4, 4]
        out = self.avgpool(out)    # shape: [batch_size, 2048, 1, 1]
        out = torch.flatten(out, start_dim=1)
        out = self.dropout_linear(out)
        out = self.linear_out(out)
        out = out.view(-1, 1, self.K[0], self.K[1])
        out = torch.sigmoid(out)
        return out
