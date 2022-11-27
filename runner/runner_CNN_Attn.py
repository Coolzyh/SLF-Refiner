import numpy as np
import os
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.utils import save_image
from model.CNN_Attn import CNN_Attn
import random
import scipy.io as sio

# from ptflops import get_model_complexity_info


# runner for CNN-attention estimator
class runner_CNN_Attn():
    def __init__(self, args):
        self.args = args
        cuda_flag = not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_flag else "cpu")
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        # torch.use_deterministic_algorithms(True)
        self.path = self.args.run + 'CNN_Attn/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def get_optimizer(self, parameters):
        if self.args.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        elif self.args.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.args.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.args.optimizer))

    def load_data(self, noise_level='all'):
        # Load training data
        training_data = sio.loadmat('./data/training_data.mat')
        RSS_minmax = sio.loadmat('./data/training_RSS_minmax.mat')
        # normalized noisy RSS measurement input (shape: [num_sample, NxPxP])
        train_RSS = training_data['RSS']
        RSS_min = RSS_minmax['RSS_min']
        RSS_max = RSS_minmax['RSS_max']
        train_RSS = (train_RSS - RSS_min) / (RSS_max - RSS_min)
        train_RSS = train_RSS.reshape(-1, self.args.N, self.args.P, self.args.P)
        # SLF image (shape: [num_sample, K0, K1])
        train_slf_img = training_data['slf_img']
        train_slf_img = train_slf_img.reshape(-1, 1, self.args.K0, self.args.K1)

        train_RSS = torch.from_numpy(train_RSS).float()
        train_slf_img = torch.from_numpy(train_slf_img).float()

        dataset = TensorDataset(train_RSS, train_slf_img)
        train_size = int(len(dataset) * 0.9)
        val_size = int(len(dataset) * 0.1)
        dataset_train, dataset_val = random_split(dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(self.args.seed))

        # Load testing data
        testing_data = sio.loadmat('./data/testing_data.mat')
        # noise level classes (shape: [num_sample, 1]) (classes: 0, 1, 2)
        noise_class = testing_data['sig_epsilon_class']
        noise_class = np.squeeze(noise_class)
        idx = (noise_class == 0) | (noise_class == 1) | (noise_class == 2)
        if noise_level == 'all':
            pass
        elif noise_level == 'low':
            idx = (noise_class == 0)
        elif noise_level == 'mid':
            idx = (noise_class == 1)
        elif noise_level == 'high':
            idx = (noise_class == 2)
        else:
            raise NotImplementedError('Noise level {} not understood.'.format(noise_level))
        # normalized noisy RSS measurement input (shape: [num_sample, NxPxP])
        test_RSS = testing_data['RSS'][idx]
        test_RSS = (test_RSS - RSS_min) / (RSS_max - RSS_min)
        test_RSS = test_RSS.reshape(-1, self.args.N, self.args.P, self.args.P)
        # SLF image (shape: [num_sample, K0, K1])
        test_slf_img = testing_data['slf_img'][idx]
        test_slf_img = test_slf_img.reshape(-1, 1, self.args.K0, self.args.K1)

        test_RSS = torch.from_numpy(test_RSS).float()
        test_slf_img = torch.from_numpy(test_slf_img).float()

        dataset_test = TensorDataset(test_RSS, test_slf_img)

        train_loader = DataLoader(
            dataset_train,
            batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(
            dataset_val,
            batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(
            dataset_test,
            batch_size=self.args.batch_size, shuffle=False, num_workers=0)
        print("Data Loaded!")
        return train_loader, val_loader, test_loader

    def train(self, model, train_loader, optimizer, epoch):
        model.train()
        train_recon_loss = 0            # total loss
        train_rmse = 0                  # rmse loss
        for batch_idx,  (input_rss, target_slf) in enumerate(train_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            optimizer.zero_grad()
            slf_recon = model(input_rss)
            loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            train_recon_loss += loss.item()

            mse = F.mse_loss(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            train_rmse += mse.item()

            loss /= input_rss.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input_rss), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                               loss.item()))

        train_recon_loss /= len(train_loader.dataset)
        train_rmse /= len(train_loader.dataset)
        train_rmse = np.sqrt(train_rmse)
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_recon_loss))
        print('====> RMSE: {:.6f}'.format(train_rmse))
        return train_recon_loss, train_rmse

    def validate(self, model, val_loader):
        model.eval()
        val_recon_loss = 0           # recon loss
        val_rmse = 0           # rmse
        for batch_idx,  (input_rss, target_slf) in enumerate(val_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            slf_recon = model(input_rss)
            recon_loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            val_recon_loss += recon_loss.item()
            mse = F.mse_loss(slf_recon, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            val_rmse += mse.item()
        val_recon_loss /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)
        val_rmse = np.sqrt(val_rmse)
        print('====> Loss: {:.6f}'.format(val_recon_loss))
        print('====> RMSE: {:.6f}'.format(val_rmse))
        return val_recon_loss, val_rmse

    def test(self, model, test_loader):
        model.eval()
        test_recon_loss = 0  # recon loss
        test_rmse = 0  # rmse for our model
        for batch_idx, (input_rss, target_slf) in enumerate(test_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            slf_recon = model(input_rss)
            recon_loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            test_recon_loss += recon_loss.item()
            mse = F.mse_loss(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            test_rmse += mse.item()
        test_recon_loss /= len(test_loader.dataset)
        test_rmse /= len(test_loader.dataset)
        test_rmse = np.sqrt(test_rmse)
        print('====> Loss: {:.6f}'.format(test_recon_loss))
        print('====> RMSE for CNN-Attn SLF estimator: {:.6f}'.format(test_rmse))
        return test_recon_loss, test_rmse

    def train_save(self):
        model = CNN_Attn(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        model = model.to(self.device)
        train_loader, val_loader, test_loader = self.load_data()
        optimizer = self.get_optimizer(model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=8, min_lr=1e-5)
        train_set = []
        val_set = []
        test_set = []
        train_loss_path = self.path + 'train_loss_' + str(self.args.n_epochs_CNN_Attn) + '.npy'
        val_loss_path = self.path + 'val_loss_' + str(self.args.n_epochs_CNN_Attn) + '.npy'
        test_loss_path = self.path + 'test_loss_' + str(self.args.n_epochs_CNN_Attn) + '.npy'
        model_path = self.path + 'CNN_Attn_' + str(self.args.n_epochs_CNN_Attn) + '.pth'
        current_val_slf_loss = np.inf
        for epoch in range(1, self.args.n_epochs_CNN_Attn+1):
            print("Epoch %d learning rate：%f" % (epoch, optimizer.param_groups[0]['lr']))
            train_recon_loss, train_rmse = self.train(model, train_loader, optimizer, epoch)
            train_set.append([train_recon_loss, train_rmse])
            print('====> Validation Loss for Epoch {:d}'.format(epoch))
            val_recon_loss, val_rmse = self.validate(model, val_loader)
            val_set.append([val_recon_loss, val_rmse])
            scheduler.step(val_recon_loss)
            print('====> Test Loss for Epoch {:d}'.format(epoch))
            test_recon_loss, test_rmse = self.test(model, test_loader)
            test_set.append([test_recon_loss, test_rmse])
            if epoch % self.args.save_freq == 0:
                ckpt_path = self.path + 'model_' + str(epoch) + '.pth'
                torch.save(model.state_dict(), ckpt_path)
                print('checkpoint{}.pth saved!'.format(epoch))

            if val_rmse < current_val_slf_loss:
                torch.save(model.state_dict(), model_path)
                current_val_slf_loss = val_rmse

        train_loss = np.asarray(train_set).reshape(-1, 2)
        val_loss = np.asarray(val_set).reshape(-1, 2)
        test_loss = np.asarray(test_set).reshape(-1, 2)
        np.save(train_loss_path, train_loss)
        np.save(val_loss_path, val_loss)
        np.save(test_loss_path, test_loss)

    def test_model(self, noise_level='all'):
        model = CNN_Attn(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))

        # macs, params = get_model_complexity_info(model, (12, 6, 6), as_strings=True,
        #                                          print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        # return

        model = model.to(self.device)
        model_path = self.path + 'CNN_Attn_' + str(self.args.n_epochs_CNN_Attn) + '.pth'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        *_, test_loader = self.load_data(noise_level)
        # test model
        slf_rmse = 0        # rmse for CNN-Attn SLF Estimator
        slf_mae = 0         # mae for CNN-Attn SLF Estimator
        for batch_idx, (input_rss, target_slf) in enumerate(test_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            slf_recon = model(input_rss)
            # plot SLF reconstruction image for image1
            n = min(input_rss.size(0), 8)
            comparison = torch.cat([target_slf.view(target_slf.size(0), 1, self.args.K0, self.args.K1)[:n],
                                    slf_recon.view(slf_recon.size(0), 1, self.args.K0, self.args.K1)[:n]])
            save_image(comparison.cpu(),
                       self.path + 'reconSLF_' + noise_level + '_' + str(batch_idx) + '.png',
                       nrow=n)
            # test SLF image RMSE
            slf_rmse += F.mse_loss(slf_recon, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)
            # test SLF image MAE
            slf_mae += F.l1_loss(slf_recon, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)

        slf_rmse /= len(test_loader.dataset)
        slf_rmse = np.sqrt(slf_rmse)
        slf_mae /= len(test_loader.dataset)
        print('====> Test Model:')
        print('====> Noise Level: ' + noise_level)
        print('====> Recon SLF rmse: {:.6f}'.format(slf_rmse))
        print('====> Recon SLF mae: {:.6f}'.format(slf_mae))