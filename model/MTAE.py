# Model for Multi-task Autoencoder
from torch import nn
import torch


# Multi-task AE
class MTAE(nn.Module):
    def __init__(self, M, P, K):
        super(MTAE, self).__init__()
        self.M = M  # The number of RF nodes
        self.N = int(M*(M-1))  # The number of links
        self.P = P  # Each node has P measurement positions
        self.K = K  # pixels of SLF image
        self.encoder = Encoder_AE(M, P)
        self.decoder = Decoder(M, P, K)

    def forward(self, x):
        embedding = self.encoder(x)
        RSS, SLF, parameters, noise_level = self.decoder(embedding)
        return RSS, SLF, parameters, noise_level


# Encoder for AutoEncoder
class Encoder_AE(nn.Module):
    def __init__(self, M, P):
        super(Encoder_AE, self).__init__()
        self.M = M  # The number of RF nodes
        self.N = int(M*(M-1))  # The number of links
        self.P = P  # Each node has P measurement positions
        # input size (CxWxH) = (NxPxP)
        self.conv1 = self.conv_layer(self.N, 32, 3, padding='same')
        self.conv2 = self.conv_layer(32, 64, 3, padding='same')
        self.conv3 = self.conv_layer(64, 128, 3, padding='same')  # shape: [batch_size, 128, P, P]
        self.conv4 = self.conv_layer(128, 128, 3, padding='valid')  # shape: [batch_size, 128, P-2, P-2]
        self.conv5 = self.conv_layer(128, self.N, 1, padding='valid', activation='Linear',
                                     batch_norm=False)  # shape: [batch_size, N, P-2, P-2]

    def conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='LeakyReLU',
                   batch_norm=True):
        layer_out = []
        bias = True
        if batch_norm:
            bias = False
        layer_out.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if batch_norm:
            layer_out.append(nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01))
        if activation == 'LeakyReLU':
            layer_out.append(nn.LeakyReLU(0.3))
        elif activation == 'ReLU':
            layer_out.append(nn.ReLU())
        elif activation == 'Linear':
            pass
        elif activation == 'Sigmoid':
            layer_out.append(nn.Sigmoid())
        else:
            raise NotImplementedError('Activation Function {} not understood.'.format(activation))
        return nn.Sequential(*layer_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


# Decoder to reconstruct RSS measurements,
# reconstruct SLF images (task 1),
# estimate parameters [b, alpha] (task 2),
# estimate noise level (task 3)
class Decoder(nn.Module):
    def __init__(self, M, P, K):
        super(Decoder, self).__init__()
        self.M = M  # The number of RF nodes
        self.N = int(M*(M-1))  # The number of links
        self.P = P  # Each node has P measurement positions
        self.K = K  # pixels of SLF image
        # input size [batch_size, N, P-2, P-2]
        # decoder1
        self.deconv1 = self.deconv_layer(self.N, 128, 1, padding=0)  # [batch_size, 128, P-2, P-2]
        self.deconv2 = self.deconv_layer(128, 128, 3, padding=0)  # [batch_size, 128, P, P]
        self.deconv3 = self.deconv_layer(128, 64, 3, padding=1)  # [batch_size, 64, P, P]
        self.deconv4 = self.deconv_layer(64, 32, 3, padding=1)  # [batch_size, 32, P, P]
        # decoder2
        self.deconv5 = self.deconv_layer(32, self.N, 3, padding=1, activation='Sigmoid', batch_norm=False)  # [batch_size, N, P, P]
        # skip-connection net
        self.conv1_sc = self.conv_layer(32, 128, 3, padding='valid')      # [batch_size, 128, P-2, P-2]
        self.dropout1_sc = nn.Dropout2d(p=0.5)
        self.conv2_sc = self.conv_layer(128, 256, 1, padding='valid', activation='Linear')   # [batch_size, 256, P-2, P-2]
        # task1 net
        self.conv1_task1 = self.conv_layer(self.N, 32, 3, padding='same')      # [batch_size, 32, P-2, P-2]
        self.conv2_task1 = self.conv_layer(32, 128, 3, padding='same')       # [batch_size, 128, P-2, P-2]
        self.conv3_task1 = self.conv_layer(128, 256, 3, padding='same', activation='Linear')      # [batch_size, 256, P-2, P-2]
        self.conv4_task1 = self.conv_layer(256, 512, 3, padding='valid')     # [batch_size, 512, P-4, P-4]
        self.conv5_task1 = self.conv_layer(512, 1024, 1, padding='valid')    # [batch_size, 1024, P-4, P-4]
        self.dropout1_task1 = nn.Dropout(p=0.5)
        self.linear1_task1 = nn.Linear(1024*(self.P-4)*(self.P-4), self.K[0]*self.K[1])   # [batch_size, K[0]*K[1]]
        # task2 net
        self.conv1_task2 = self.conv_layer(self.N, 32, 3, padding='same')   # [batch_size, 32, P-2, P-2]
        self.conv2_task2 = self.conv_layer(32, 128, 3, padding='same')    # [batch_size, 128, P-2, P-2]
        self.conv3_task2 = self.conv_layer(128, 256, 3, padding='valid')  # [batch_size, 256, P-4, P-4]
        self.dropout1_task2 = nn.Dropout(p=0.5)
        self.linear1_task2 = nn.Linear(256 * (self.P - 4) * (self.P - 4), self.N+1)  # [batch_size, N+1]
        # task3 net
        self.conv1_task3 = self.conv_layer(self.N, 32, 3, padding='same')  # [batch_size, 32, P-2, P-2]
        self.conv2_task3 = self.conv_layer(32, 128, 3, padding='same')  # [batch_size, 128, P-2, P-2]
        self.conv3_task3 = self.conv_layer(128, 256, 3, padding='valid')  # [batch_size, 256, P-4, P-4]
        self.dropout1_task3 = nn.Dropout(p=0.5)
        self.linear1_task3 = nn.Linear(256 * (self.P - 4) * (self.P - 4), 3)  # [batch_size, 3]

    def conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='LeakyReLU',
                   batch_norm=True):
        layer_out = []
        bias = True
        if batch_norm:
            bias = False
        layer_out.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if batch_norm:
            layer_out.append(nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01))
        if activation == 'LeakyReLU':
            layer_out.append(nn.LeakyReLU(0.3))
        elif activation == 'ReLU':
            layer_out.append(nn.ReLU())
        elif activation == 'Linear':
            pass
        elif activation == 'Sigmoid':
            layer_out.append(nn.Sigmoid())
        else:
            raise NotImplementedError('Activation Function {} not understood.'.format(activation))
        return nn.Sequential(*layer_out)

    def deconv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='LeakyReLU',
                     batch_norm=True):
        layer_out = []
        bias = True
        if batch_norm:
            bias = False
        layer_out.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if batch_norm:
            layer_out.append(nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01))
        if activation == 'LeakyReLU':
            layer_out.append(nn.LeakyReLU(0.3))
        elif activation == 'ReLU':
            layer_out.append(nn.ReLU())
        elif activation == 'Linear':
            pass
        elif activation == 'Sigmoid':
            layer_out.append(nn.Sigmoid())
        else:
            raise NotImplementedError('Activation Function {} not understood.'.format(activation))
        return nn.Sequential(*layer_out)

    def decoder1(self, x):
        # x: [batch_size, N, P-2, P-2]
        out = self.deconv1(x)  # [batch_size, 128, P-2, P-2]
        out = self.deconv2(out)  # [batch_size, 128, P, P]
        out = self.deconv3(out)  # [batch_size, 64, P, P]
        out = self.deconv4(out)  # [batch_size, 32, P, P]
        return out

    def decoder2(self, x):
        # x: [batch_size, 32, P, P]
        out = self.deconv5(x)  # [batch_size, N, P, P]
        return out

    # skip connection net
    def SC_net(self, x):
        # x: [batch_size, 32, P, P]
        out = self.conv1_sc(x)      # [batch_size, 128, P-2, P-2]
        out = self.dropout1_sc(out)
        out = self.conv2_sc(out)   # [batch_size, 256, P-2, P-2]
        return out

    # reconstruct SLF images net (task 1)
    def task1_net(self, x, sc):
        # x: encoder embedding: [batch_size, N, P-2, P-2]
        # sc: skip connection output: [batch_size, 256, P-2, P-2]
        out = self.conv1_task1(x)      # [batch_size, 32, P-2, P-2]
        out = self.conv2_task1(out)       # [batch_size, 128, P-2, P-2]
        out = self.conv3_task1(out)      # [batch_size, 256, P-2, P-2]
        out = nn.LeakyReLU(0.3)(out+sc)
        out = self.conv4_task1(out)     # [batch_size, 512, P-4, P-4]
        out = self.conv5_task1(out)    # [batch_size, 1024, P-4, P-4]
        out = torch.flatten(out, start_dim=1)                        # [batch_size, 1024*(P-4)*(P-4)]
        out = self.dropout1_task1(out)
        out = self.linear1_task1(out)    # [batch_size, K[0]*K[1]]
        out = torch.sigmoid(out)
        return out

    # estimate parameters [b, alpha] (task 2)
    def task2_net(self, x):
        # x: encoder embedding: [batch_size, N, P-2, P-2]
        out = self.conv1_task2(x)   # [batch_size, 32, P-2, P-2]
        out = self.conv2_task2(out)    # [batch_size, 128, P-2, P-2]
        out = self.conv3_task2(out)  # [batch_size, 256, P-4, P-4]
        out = torch.flatten(out, start_dim=1)  # [batch_size, 256*(P-4)*(P-4)]
        out = self.dropout1_task2(out)
        out = self.linear1_task2(out)  # [batch_size, N+1]
        out = torch.sigmoid(out)
        return out

    # estimate noise level (task 3)
    def task3_net(self, x):
        # x: encoder embedding: [batch_size, N, P-2, P-2]
        out = self.conv1_task3(x)  # [batch_size, 32, P-2, P-2]
        out = self.conv2_task3(out)  # [batch_size, 128, P-2, P-2]
        out = self.conv3_task3(out)  # [batch_size, 256, P-4, P-4]
        out = torch.flatten(out, start_dim=1)  # [batch_size, 256*(P-4)*(P-4)]
        out = self.dropout1_task3(out)
        out = self.linear1_task3(out)  # [batch_size, 3]
        return out

    def forward(self, x):
        # x: encoder embedding: [batch_size, N, P-2, P-2]
        # reconstruct RSS measurements
        sc_in = self.decoder1(x)
        RSS = self.decoder2(sc_in)
        # reconstruct SLF images (task 1)
        sc_out = self.SC_net(sc_in)
        SLF = self.task1_net(x, sc_out)
        # estimate parameters [b, alpha] (task 2)
        parameters = self.task2_net(x)
        # estimate noise level (task 3)
        noise_level = self.task3_net(x)
        return RSS, SLF, parameters, noise_level