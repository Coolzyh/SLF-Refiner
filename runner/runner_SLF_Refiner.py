import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.utils import save_image
from model.SLF_Refiner import SLF_Refiner
from model.CNN_Attn import CNN_Attn
from model.MTAE import MTAE
import random
import scipy.io as sio
# from ptflops import get_model_complexity_info


# Define regularization term
# Total variation
class TVLoss(nn.Module):
    def __init__(self, weight: float = 1) -> None:
        """Total Variation Loss

        Args:
            weight (float): weight of TV loss
        """
        super().__init__()
        self.weight = weight

    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        # return sum TV loss of the whole batch
        return self.weight * (tv_h + tv_w) / (c * h * w)


# runner for SLF-Refiner
class runner_SLF_Refiner():
    def __init__(self, args):
        self.args = args
        cuda_flag = not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_flag else "cpu")
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        # torch.use_deterministic_algorithms(True)
        self.path = self.args.run + 'SLF_Refiner/'
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

        # SLF image ENR (shape: [num_sample, K0, K1])
        train_slf_img_ENR = training_data['slf_img_ENR']
        train_slf_img_ENR = train_slf_img_ENR.reshape(-1, 1, self.args.K0, self.args.K1)

        train_RSS = torch.from_numpy(train_RSS).float()
        train_slf_img = torch.from_numpy(train_slf_img).float()
        train_slf_img_ENR = torch.from_numpy(train_slf_img_ENR).float()

        dataset = TensorDataset(train_RSS, train_slf_img_ENR, train_slf_img)
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

        # SLF image ENR (shape: [num_sample, K0, K1])
        test_slf_img_ENR = testing_data['slf_img_ENR'][idx]
        test_slf_img_ENR = test_slf_img_ENR.reshape(-1, 1, self.args.K0, self.args.K1)

        test_RSS = torch.from_numpy(test_RSS).float()
        test_slf_img = torch.from_numpy(test_slf_img).float()
        test_slf_img_ENR = torch.from_numpy(test_slf_img_ENR).float()

        dataset_test = TensorDataset(test_RSS, test_slf_img_ENR, test_slf_img)

        train_loader = DataLoader(
            dataset_train,
            batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(
            dataset_val,
            batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(
            dataset_test,
            batch_size=self.args.batch_size, shuffle=False)
        print("Data Loaded!")
        return train_loader, val_loader, test_loader

    # train SLF Refiner for ENR estimation
    def train(self, model, train_loader, optimizer, epoch):
        model.train()
        train_loss = 0                  # total loss
        train_rmse1 = 0                 # rmse for ENR
        train_rmse2 = 0                 # rmse for SLF refiner
        for batch_idx,  (input_rss, slf_coarse, target_slf) in enumerate(train_loader):
            input_rss = input_rss.to(self.device)
            slf_coarse = slf_coarse.to(self.device)
            target_slf = target_slf.to(self.device)
            optimizer.zero_grad()
            slf_recon = model(input_rss, slf_coarse)
            loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum')/(self.args.K0*self.args.K1)

            # # regularize term
            # reg_tv = TVLoss(weight=0.1)
            # reg_loss = reg_tv(slf_recon)
            # loss += reg_loss

            train_loss += loss.item()

            mse1 = F.mse_loss(slf_coarse, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            mse2 = F.mse_loss(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            train_rmse1 += mse1.item()
            train_rmse2 += mse2.item()

            loss /= input_rss.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input_rss), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                               loss.item()))

        train_loss /= len(train_loader.dataset)
        train_rmse1 /= len(train_loader.dataset)
        train_rmse1 = np.sqrt(train_rmse1)
        train_rmse2 /= len(train_loader.dataset)
        train_rmse2 = np.sqrt(train_rmse2)
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss))
        print('====> RMSE for ENR: {:.6f}'.format(train_rmse1))
        print('====> RMSE for SLF Refiner: {:.6f}'.format(train_rmse2))
        return train_loss, train_rmse1, train_rmse2

    def validate(self, model, val_loader):
        model.eval()
        val_loss = 0                 # recon loss
        val_rmse1 = 0                # rmse for ENR
        val_rmse2 = 0                # rmse for our model
        for batch_idx,  (input_rss, slf_coarse, target_slf) in enumerate(val_loader):
            input_rss = input_rss.to(self.device)
            slf_coarse = slf_coarse.to(self.device)
            target_slf = target_slf.to(self.device)
            slf_recon = model(input_rss, slf_coarse)
            loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            val_loss += loss.item()
            mse1 = F.mse_loss(slf_coarse, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            mse2 = F.mse_loss(slf_recon, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            val_rmse1 += mse1.item()
            val_rmse2 += mse2.item()
        val_loss /= len(val_loader.dataset)
        val_rmse1 /= len(val_loader.dataset)
        val_rmse1 = np.sqrt(val_rmse1)
        val_rmse2 /= len(val_loader.dataset)
        val_rmse2 = np.sqrt(val_rmse2)
        print('====> Recon loss: {:.6f}'.format(val_loss))
        print('====> RMSE for ENR: {:.6f}'.format(val_rmse1))
        print('====> RMSE for SLF Refiner: {:.6f}'.format(val_rmse2))
        return val_loss, val_rmse1, val_rmse2

    def test(self, model, test_loader):
        model.eval()
        test_loss = 0  # recon loss
        test_rmse1 = 0  # rmse for ENR
        test_rmse2 = 0  # rmse for our model
        for batch_idx, (input_rss, slf_coarse, target_slf) in enumerate(test_loader):
            input_rss = input_rss.to(self.device)
            slf_coarse = slf_coarse.to(self.device)
            target_slf = target_slf.to(self.device)
            slf_recon = model(input_rss, slf_coarse)
            loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            test_loss += loss.item()
            mse1 = F.mse_loss(slf_coarse, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            mse2 = F.mse_loss(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            test_rmse1 += mse1.item()
            test_rmse2 += mse2.item()
        test_loss /= len(test_loader.dataset)
        test_rmse1 /= len(test_loader.dataset)
        test_rmse1 = np.sqrt(test_rmse1)
        test_rmse2 /= len(test_loader.dataset)
        test_rmse2 = np.sqrt(test_rmse2)
        print('====> Recon loss: {:.6f}'.format(test_loss))
        print('====> RMSE for ENR: {:.6f}'.format(test_rmse1))
        print('====> RMSE for SLF Refiner: {:.6f}'.format(test_rmse2))
        return test_loss, test_rmse1, test_rmse2

    def train_save(self):
        print("Start training SLF refiner for ENR estimator!")
        model = SLF_Refiner(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        model = model.to(self.device)
        train_loader, val_loader, test_loader = self.load_data()
        optimizer = self.get_optimizer(model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=8, min_lr=1e-5)
        train_set = []
        val_set = []
        test_set = []
        path_save = self.path + 'ENR/'
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        train_loss_path = path_save + 'train_loss_' + str(self.args.n_epochs_Refiner) + '.npy'
        val_loss_path = path_save + 'val_loss_' + str(self.args.n_epochs_Refiner) + '.npy'
        test_loss_path = path_save + 'test_loss_' + str(self.args.n_epochs_Refiner) + '.npy'
        model_path = path_save + 'SLF_Refiner_' + str(self.args.n_epochs_Refiner) + '.pth'
        current_val_slf_loss = np.inf
        for epoch in range(1, self.args.n_epochs_Refiner+1):
            print("Epoch %d learning rate：%f" % (epoch, optimizer.param_groups[0]['lr']))
            train_loss, train_rmse1, train_rmse2 = self.train(model, train_loader, optimizer, epoch)
            train_set.append([train_loss, train_rmse1, train_rmse2])
            print('====> Validation Loss for Epoch {:d}'.format(epoch))
            val_loss, val_rmse1, val_rmse2 = self.validate(model, val_loader)
            val_set.append([val_loss, val_rmse1, val_rmse2])
            scheduler.step(val_loss)
            print('====> Test Loss for Epoch {:d}'.format(epoch))
            test_loss, test_rmse1, test_rmse2 = self.test(model, test_loader)
            test_set.append([test_loss, test_rmse1, test_rmse2])
            if epoch % self.args.save_freq == 0:
                ckpt_path = path_save + 'model_' + str(epoch) + '.pth'
                torch.save(model.state_dict(), ckpt_path)
                print('checkpoint{}.pth saved!'.format(epoch))

            if val_rmse2 < current_val_slf_loss:
                torch.save(model.state_dict(), model_path)
                current_val_slf_loss = val_rmse2

        train_loss = np.asarray(train_set).reshape(-1, 3)
        val_loss = np.asarray(val_set).reshape(-1, 3)
        test_loss = np.asarray(test_set).reshape(-1, 3)
        np.save(train_loss_path, train_loss)
        np.save(val_loss_path, val_loss)
        np.save(test_loss_path, test_loss)

    def test_model(self, noise_level='all'):
        print("Start testing SLF refiner for ENR estimator!")
        model = SLF_Refiner(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))

        # macs, params = get_model_complexity_info(model.generator, (12, 6, 6), as_strings=True,
        #                                          print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        # return

        model = model.to(self.device)
        path_save = self.path + 'ENR/'
        model_path = path_save + 'SLF_Refiner_' + str(self.args.n_epochs_Refiner) + '.pth'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        *_, test_loader = self.load_data(noise_level)
        # test model
        slf_rmse1 = 0        # rmse for stage 1
        slf_mae1 = 0         # mae for stage 1
        slf_rmse2 = 0        # rmse for stage 2
        slf_mae2 = 0         # mae for stage 2
        for batch_idx, (input_rss, slf_coarse, target_slf) in enumerate(test_loader):
            input_rss = input_rss.to(self.device)
            slf_coarse = slf_coarse.to(self.device)
            target_slf = target_slf.to(self.device)
            slf_recon = model(input_rss, slf_coarse)
            # save one image for paper show
            if batch_idx == 3:
                clean_slf = target_slf.view(target_slf.size(0), self.args.K0, self.args.K1)[0].detach().cpu().numpy()
                enr_slf = slf_coarse.view(slf_coarse.size(0), self.args.K0, self.args.K1)[0].detach().cpu().numpy()
                refine_slf = slf_recon.view(slf_recon.size(0), self.args.K0, self.args.K1)[0].detach().cpu().numpy()
                np.save('results/compare/' + 'slf_target_sample.npy', clean_slf)
                np.save('results/compare/'+'slf_enr_sample.npy', enr_slf)
                np.save('results/compare/' + 'slf_enr_refine_sample.npy', refine_slf)

            # plot SLF reconstruction image
            n = min(input_rss.size(0), 8)
            comparison = torch.cat([target_slf.view(target_slf.size(0), 1, self.args.K0, self.args.K1)[:n],
                                    slf_coarse.view(slf_coarse.size(0), 1, self.args.K0, self.args.K1)[:n],
                                    slf_recon.view(slf_recon.size(0), 1, self.args.K0, self.args.K1)[:n]])
            save_image(comparison.cpu(),
                       path_save + 'reconSLF_' + noise_level + '_' + str(batch_idx) + '.png',
                       nrow=n)
            # test SLF image RMSE
            slf_rmse1 += F.mse_loss(slf_coarse, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)
            slf_rmse2 += F.mse_loss(slf_recon, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)
            # test SLF image MAE
            slf_mae1 += F.l1_loss(slf_coarse, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)
            slf_mae2 += F.l1_loss(slf_recon, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)

        slf_rmse1 /= len(test_loader.dataset)
        slf_rmse1 = np.sqrt(slf_rmse1)
        slf_rmse2 /= len(test_loader.dataset)
        slf_rmse2 = np.sqrt(slf_rmse2)
        slf_mae1 /= len(test_loader.dataset)
        slf_mae2 /= len(test_loader.dataset)
        print('====> Test Model:')
        print('====> Noise Level: ' + noise_level)
        print('====> ENR SLF rmse: {:.6f}'.format(slf_rmse1))
        print('====> Recon SLF rmse: {:.6f}'.format(slf_rmse2))
        print('====> ENR SLF mae: {:.6f}'.format(slf_mae1))
        print('====> Recon SLF mae: {:.6f}'.format(slf_mae2))

    # train SLF Refiner for well-trained CNN_Attn SLF estimator
    def train_with_cnn_attn(self, model, coarse_estimator, train_loader, optimizer, epoch):
        model.train()
        coarse_estimator.eval()         # the well-trained SLF estimator for first stage
        train_loss = 0                  # total loss
        train_rmse1 = 0                 # rmse for first stage estimation
        train_rmse2 = 0                 # rmse for SLF refiner
        for batch_idx,  (input_rss, _, target_slf) in enumerate(train_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            optimizer.zero_grad()
            slf_coarse = coarse_estimator(input_rss)
            slf_recon = model(input_rss, slf_coarse)
            loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            train_loss += loss.item()

            mse1 = F.mse_loss(slf_coarse, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            mse2 = F.mse_loss(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            train_rmse1 += mse1.item()
            train_rmse2 += mse2.item()

            loss /= input_rss.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input_rss), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                               loss.item()))

        train_loss /= len(train_loader.dataset)
        train_rmse1 /= len(train_loader.dataset)
        train_rmse1 = np.sqrt(train_rmse1)
        train_rmse2 /= len(train_loader.dataset)
        train_rmse2 = np.sqrt(train_rmse2)
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss))
        print('====> RMSE for first stage estimation: {:.6f}'.format(train_rmse1))
        print('====> RMSE for SLF Refiner: {:.6f}'.format(train_rmse2))
        return train_loss, train_rmse1, train_rmse2

    def validate_with_cnn_attn(self, model, coarse_estimator, val_loader):
        model.eval()
        coarse_estimator.eval()
        val_loss = 0                 # recon loss
        val_rmse1 = 0                # rmse for first stage SLF estimator
        val_rmse2 = 0                # rmse for our model
        for batch_idx,  (input_rss, _, target_slf) in enumerate(val_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            slf_coarse = coarse_estimator(input_rss)
            slf_recon = model(input_rss, slf_coarse)
            loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            val_loss += loss.item()
            mse1 = F.mse_loss(slf_coarse, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            mse2 = F.mse_loss(slf_recon, target_slf, reduction='sum')/(self.args.K0*self.args.K1)
            val_rmse1 += mse1.item()
            val_rmse2 += mse2.item()
        val_loss /= len(val_loader.dataset)
        val_rmse1 /= len(val_loader.dataset)
        val_rmse1 = np.sqrt(val_rmse1)
        val_rmse2 /= len(val_loader.dataset)
        val_rmse2 = np.sqrt(val_rmse2)
        print('====> Recon loss: {:.6f}'.format(val_loss))
        print('====> RMSE for first stage estimation: {:.6f}'.format(val_rmse1))
        print('====> RMSE for SLF Refiner: {:.6f}'.format(val_rmse2))
        return val_loss, val_rmse1, val_rmse2

    def test_with_cnn_attn(self, model, coarse_estimator, test_loader):
        model.eval()
        coarse_estimator.eval()
        test_loss = 0  # recon loss
        test_rmse1 = 0  # rmse for first stage SLF estimator
        test_rmse2 = 0  # rmse for our model
        for batch_idx, (input_rss, _, target_slf) in enumerate(test_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            with torch.no_grad():
                slf_coarse = coarse_estimator(input_rss)
            slf_recon = model(input_rss, slf_coarse)
            loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            test_loss += loss.item()
            mse1 = F.mse_loss(slf_coarse, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            mse2 = F.mse_loss(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            test_rmse1 += mse1.item()
            test_rmse2 += mse2.item()
        test_loss /= len(test_loader.dataset)
        test_rmse1 /= len(test_loader.dataset)
        test_rmse1 = np.sqrt(test_rmse1)
        test_rmse2 /= len(test_loader.dataset)
        test_rmse2 = np.sqrt(test_rmse2)
        print('====> Recon loss: {:.6f}'.format(test_loss))
        print('====> RMSE for first stage estimation: {:.6f}'.format(test_rmse1))
        print('====> RMSE for SLF Refiner: {:.6f}'.format(test_rmse2))
        return test_loss, test_rmse1, test_rmse2

    def train_save_with_cnn_attn(self):
        print("Start training SLF refiner for CNN-Attn SLF estimator!")
        model = SLF_Refiner(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        model = model.to(self.device)
        path_save = self.path + 'CNN_Attn/'
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        coarse_estimator = CNN_Attn(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        coarse_estimator = coarse_estimator.to(self.device)
        coarse_estimator_path = path_save + 'CNN_Attn_' + str(self.args.n_epochs_CNN_Attn) + '.pth'
        coarse_estimator.load_state_dict(torch.load(coarse_estimator_path, map_location=self.device))
        coarse_estimator.eval()
        for param in coarse_estimator.parameters():
            param.requires_grad = False

        train_loader, val_loader, test_loader = self.load_data()
        optimizer = self.get_optimizer(model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=8, min_lr=1e-5)
        train_set = []
        val_set = []
        test_set = []
        train_loss_path = path_save + 'train_loss_' + str(self.args.n_epochs_Refiner) + '.npy'
        val_loss_path = path_save + 'val_loss_' + str(self.args.n_epochs_Refiner) + '.npy'
        test_loss_path = path_save + 'test_loss_' + str(self.args.n_epochs_Refiner) + '.npy'
        model_path = path_save + 'SLF_Refiner_' + str(self.args.n_epochs_Refiner) + '.pth'
        current_val_slf_loss = np.inf
        for epoch in range(1, self.args.n_epochs_Refiner + 1):
            print("Epoch %d learning rate：%f" % (epoch, optimizer.param_groups[0]['lr']))
            train_loss, train_rmse1, train_rmse2 = self.train_with_cnn_attn(model, coarse_estimator, train_loader, optimizer, epoch)
            train_set.append([train_loss, train_rmse1, train_rmse2])
            print('====> Validation Loss for Epoch {:d}'.format(epoch))
            val_loss, val_rmse1, val_rmse2 = self.validate_with_cnn_attn(model, coarse_estimator, val_loader)
            val_set.append([val_loss, val_rmse1, val_rmse2])
            scheduler.step(val_loss)
            print('====> Test Loss for Epoch {:d}'.format(epoch))
            test_loss, test_rmse1, test_rmse2 = self.test_with_cnn_attn(model, coarse_estimator, test_loader)
            test_set.append([test_loss, test_rmse1, test_rmse2])
            if epoch % self.args.save_freq == 0:
                ckpt_path = path_save + 'model_' + str(epoch) + '.pth'
                torch.save(model.state_dict(), ckpt_path)
                print('checkpoint{}.pth saved!'.format(epoch))

            if val_rmse2 < current_val_slf_loss:
                torch.save(model.state_dict(), model_path)
                current_val_slf_loss = val_rmse2

        train_loss = np.asarray(train_set).reshape(-1, 3)
        val_loss = np.asarray(val_set).reshape(-1, 3)
        test_loss = np.asarray(test_set).reshape(-1, 3)
        np.save(train_loss_path, train_loss)
        np.save(val_loss_path, val_loss)
        np.save(test_loss_path, test_loss)

    def test_model_with_cnn_attn(self, noise_level='all'):
        print("Start testing SLF refiner for CNN-Attn SLF estimator!")
        model = SLF_Refiner(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        model = model.to(self.device)
        coarse_estimator = CNN_Attn(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        coarse_estimator = coarse_estimator.to(self.device)
        path_save = self.path + 'CNN_Attn/'
        model_path = path_save + 'SLF_Refiner_' + str(self.args.n_epochs_Refiner) + '.pth'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        coarse_estimator_path = path_save + 'CNN_Attn_' + str(self.args.n_epochs_CNN_Attn) + '.pth'
        coarse_estimator.load_state_dict(torch.load(coarse_estimator_path, map_location=self.device))
        coarse_estimator.eval()
        *_, test_loader = self.load_data(noise_level)
        # test model
        slf_rmse1 = 0  # rmse for stage 1
        slf_mae1 = 0  # mae for stage 1
        slf_rmse2 = 0  # rmse for stage 2
        slf_mae2 = 0  # mae for stage 2
        for batch_idx, (input_rss, _, target_slf) in enumerate(test_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            slf_coarse = coarse_estimator(input_rss)
            slf_recon = model(input_rss, slf_coarse)

            # # save one image for paper show
            # if batch_idx == 2:
            #     attcnn_slf = slf_coarse.view(slf_coarse.size(0), self.args.K0, self.args.K1)[6].detach().cpu().numpy()
            #     refine_slf = slf_recon.view(slf_recon.size(0), self.args.K0, self.args.K1)[6].detach().cpu().numpy()
            #     np.save('results/compare/' + 'slf_attcnn_sample.npy', attcnn_slf)
            #     np.save('results/compare/' + 'slf_attcnn_refine_sample.npy', refine_slf)

            # plot SLF reconstruction image
            n = min(input_rss.size(0), 8)
            comparison = torch.cat([target_slf.view(target_slf.size(0), 1, self.args.K0, self.args.K1)[:n],
                                    slf_coarse.view(slf_coarse.size(0), 1, self.args.K0, self.args.K1)[:n],
                                    slf_recon.view(slf_recon.size(0), 1, self.args.K0, self.args.K1)[:n]])
            save_image(comparison.cpu(),
                       path_save + 'reconSLF_' + noise_level + '_' + str(batch_idx) + '.png',
                       nrow=n)
            # test SLF image RMSE
            slf_rmse1 += F.mse_loss(slf_coarse, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)
            slf_rmse2 += F.mse_loss(slf_recon, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)
            # test SLF image MAE
            slf_mae1 += F.l1_loss(slf_coarse, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)
            slf_mae2 += F.l1_loss(slf_recon, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)

        slf_rmse1 /= len(test_loader.dataset)
        slf_rmse1 = np.sqrt(slf_rmse1)
        slf_rmse2 /= len(test_loader.dataset)
        slf_rmse2 = np.sqrt(slf_rmse2)
        slf_mae1 /= len(test_loader.dataset)
        slf_mae2 /= len(test_loader.dataset)
        print('====> Test Model:')
        print('====> Noise Level: ' + noise_level)
        print('====> First stage SLF rmse: {:.6f}'.format(slf_rmse1))
        print('====> Refined SLF rmse: {:.6f}'.format(slf_rmse2))
        print('====> First stage SLF mae: {:.6f}'.format(slf_mae1))
        print('====> Refined SLF mae: {:.6f}'.format(slf_mae2))

    # train SLF Refiner for well-trained MTAE SLF estimator
    def train_with_mtae(self, model, coarse_estimator, train_loader, optimizer, epoch):
        model.train()
        coarse_estimator.eval()  # the well-trained SLF estimator for first stage
        train_loss = 0  # total loss
        train_rmse1 = 0  # rmse for first stage estimation
        train_rmse2 = 0  # rmse for SLF refiner
        for batch_idx, (input_rss, _, target_slf) in enumerate(train_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            optimizer.zero_grad()
            with torch.no_grad():
                _, slf_coarse, *_ = coarse_estimator(input_rss)
            slf_coarse = torch.reshape(slf_coarse, (-1, 1, self.args.K0, self.args.K1))
            slf_recon = model(input_rss, slf_coarse)
            loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            train_loss += loss.item()

            mse1 = F.mse_loss(slf_coarse, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            mse2 = F.mse_loss(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            train_rmse1 += mse1.item()
            train_rmse2 += mse2.item()

            loss /= input_rss.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input_rss), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        loss.item()))

        train_loss /= len(train_loader.dataset)
        train_rmse1 /= len(train_loader.dataset)
        train_rmse1 = np.sqrt(train_rmse1)
        train_rmse2 /= len(train_loader.dataset)
        train_rmse2 = np.sqrt(train_rmse2)
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss))
        print('====> RMSE for first stage estimation: {:.6f}'.format(train_rmse1))
        print('====> RMSE for SLF Refiner: {:.6f}'.format(train_rmse2))
        return train_loss, train_rmse1, train_rmse2

    def validate_with_mtae(self, model, coarse_estimator, val_loader):
        model.eval()
        coarse_estimator.eval()
        val_loss = 0  # recon loss
        val_rmse1 = 0  # rmse for first stage SLF estimator
        val_rmse2 = 0  # rmse for our model
        for batch_idx, (input_rss, _, target_slf) in enumerate(val_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            _, slf_coarse, *_ = coarse_estimator(input_rss)
            slf_coarse = torch.reshape(slf_coarse, (-1, 1, self.args.K0, self.args.K1))
            slf_recon = model(input_rss, slf_coarse)
            loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            val_loss += loss.item()
            mse1 = F.mse_loss(slf_coarse, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            mse2 = F.mse_loss(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            val_rmse1 += mse1.item()
            val_rmse2 += mse2.item()
        val_loss /= len(val_loader.dataset)
        val_rmse1 /= len(val_loader.dataset)
        val_rmse1 = np.sqrt(val_rmse1)
        val_rmse2 /= len(val_loader.dataset)
        val_rmse2 = np.sqrt(val_rmse2)
        print('====> Recon loss: {:.6f}'.format(val_loss))
        print('====> RMSE for first stage estimation: {:.6f}'.format(val_rmse1))
        print('====> RMSE for SLF Refiner: {:.6f}'.format(val_rmse2))
        return val_loss, val_rmse1, val_rmse2

    def test_with_mtae(self, model, coarse_estimator, test_loader):
        model.eval()
        coarse_estimator.eval()
        test_loss = 0  # recon loss
        test_rmse1 = 0  # rmse for first stage SLF estimator
        test_rmse2 = 0  # rmse for our model
        for batch_idx, (input_rss, _, target_slf) in enumerate(test_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            _, slf_coarse, *_ = coarse_estimator(input_rss)
            slf_coarse = torch.reshape(slf_coarse, (-1, 1, self.args.K0, self.args.K1))
            slf_recon = model(input_rss, slf_coarse)
            loss = F.binary_cross_entropy(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            test_loss += loss.item()
            mse1 = F.mse_loss(slf_coarse, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            mse2 = F.mse_loss(slf_recon, target_slf, reduction='sum') / (self.args.K0 * self.args.K1)
            test_rmse1 += mse1.item()
            test_rmse2 += mse2.item()
        test_loss /= len(test_loader.dataset)
        test_rmse1 /= len(test_loader.dataset)
        test_rmse1 = np.sqrt(test_rmse1)
        test_rmse2 /= len(test_loader.dataset)
        test_rmse2 = np.sqrt(test_rmse2)
        print('====> Recon loss: {:.6f}'.format(test_loss))
        print('====> RMSE for first stage estimation: {:.6f}'.format(test_rmse1))
        print('====> RMSE for SLF Refiner: {:.6f}'.format(test_rmse2))
        return test_loss, test_rmse1, test_rmse2

    def train_save_with_mtae(self):
        print("Start training SLF refiner for MTAE SLF estimator!")
        model = SLF_Refiner(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        model = model.to(self.device)
        path_save = self.path + 'MTAE/'
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        coarse_estimator = MTAE(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        coarse_estimator = coarse_estimator.to(self.device)
        coarse_estimator_path = path_save + 'MTAE_' + str(self.args.n_epochs_MTAE) + '.pth'
        coarse_estimator.load_state_dict(torch.load(coarse_estimator_path, map_location=self.device))
        coarse_estimator.eval()
        for param in coarse_estimator.parameters():
            param.requires_grad = False

        train_loader, val_loader, test_loader = self.load_data()
        optimizer = self.get_optimizer(model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=8, min_lr=1e-5)
        train_set = []
        val_set = []
        test_set = []
        train_loss_path = path_save + 'train_loss_' + str(self.args.n_epochs_Refiner) + '.npy'
        val_loss_path = path_save + 'val_loss_' + str(self.args.n_epochs_Refiner) + '.npy'
        test_loss_path = path_save + 'test_loss_' + str(self.args.n_epochs_Refiner) + '.npy'
        model_path = path_save + 'SLF_Refiner_' + str(self.args.n_epochs_Refiner) + '.pth'
        current_val_slf_loss = np.inf
        for epoch in range(1, self.args.n_epochs_Refiner + 1):
            print("Epoch %d learning rate：%f" % (epoch, optimizer.param_groups[0]['lr']))
            train_loss, train_rmse1, train_rmse2 = self.train_with_mtae(model, coarse_estimator, train_loader,
                                                                            optimizer, epoch)
            train_set.append([train_loss, train_rmse1, train_rmse2])
            print('====> Validation Loss for Epoch {:d}'.format(epoch))
            val_loss, val_rmse1, val_rmse2 = self.validate_with_mtae(model, coarse_estimator, val_loader)
            val_set.append([val_loss, val_rmse1, val_rmse2])
            scheduler.step(val_loss)
            print('====> Test Loss for Epoch {:d}'.format(epoch))
            test_loss, test_rmse1, test_rmse2 = self.test_with_mtae(model, coarse_estimator, test_loader)
            test_set.append([test_loss, test_rmse1, test_rmse2])
            if epoch % self.args.save_freq == 0:
                ckpt_path = path_save + 'model_' + str(epoch) + '.pth'
                torch.save(model.state_dict(), ckpt_path)
                print('checkpoint{}.pth saved!'.format(epoch))

            if val_rmse2 < current_val_slf_loss:
                torch.save(model.state_dict(), model_path)
                current_val_slf_loss = val_rmse2

        train_loss = np.asarray(train_set).reshape(-1, 3)
        val_loss = np.asarray(val_set).reshape(-1, 3)
        test_loss = np.asarray(test_set).reshape(-1, 3)
        np.save(train_loss_path, train_loss)
        np.save(val_loss_path, val_loss)
        np.save(test_loss_path, test_loss)

    def test_model_with_mtae(self, noise_level='all'):
        print("Start testing SLF refiner for MTAE SLF estimator!")
        model = SLF_Refiner(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        model = model.to(self.device)
        coarse_estimator = MTAE(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        coarse_estimator = coarse_estimator.to(self.device)
        path_save = self.path + 'MTAE/'
        model_path = path_save + 'SLF_Refiner_' + str(self.args.n_epochs_Refiner) + '.pth'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        coarse_estimator_path = path_save + 'MTAE_' + str(self.args.n_epochs_MTAE) + '.pth'
        coarse_estimator.load_state_dict(torch.load(coarse_estimator_path, map_location=self.device))
        coarse_estimator.eval()
        *_, test_loader = self.load_data(noise_level)
        # test model
        slf_rmse1 = 0  # rmse for stage 1
        slf_mae1 = 0  # mae for stage 1
        slf_rmse2 = 0  # rmse for stage 2
        slf_mae2 = 0  # mae for stage 2
        for batch_idx, (input_rss, _, target_slf) in enumerate(test_loader):
            input_rss = input_rss.to(self.device)
            target_slf = target_slf.to(self.device)
            _, slf_coarse, *_ = coarse_estimator(input_rss)
            slf_coarse = torch.reshape(slf_coarse, (-1, 1, self.args.K0, self.args.K1))
            slf_recon = model(input_rss, slf_coarse)

            # # save one image for paper show
            # if batch_idx == 2:
            #     mtae_slf = slf_coarse.view(slf_coarse.size(0), self.args.K0, self.args.K1)[6].detach().cpu().numpy()
            #     refine_slf = slf_recon.view(slf_recon.size(0), self.args.K0, self.args.K1)[6].detach().cpu().numpy()
            #     np.save('results/compare/' + 'slf_mtae_sample.npy', mtae_slf)
            #     np.save('results/compare/' + 'slf_mtae_refine_sample.npy', refine_slf)

            # plot SLF reconstruction image
            n = min(input_rss.size(0), 8)
            comparison = torch.cat([target_slf.view(target_slf.size(0), 1, self.args.K0, self.args.K1)[:n],
                                    slf_coarse.view(slf_coarse.size(0), 1, self.args.K0, self.args.K1)[:n],
                                    slf_recon.view(slf_recon.size(0), 1, self.args.K0, self.args.K1)[:n]])
            save_image(comparison.cpu(),
                       path_save + 'reconSLF_' + noise_level + '_' + str(batch_idx) + '.png',
                       nrow=n)
            # test SLF image RMSE
            slf_rmse1 += F.mse_loss(slf_coarse, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)
            slf_rmse2 += F.mse_loss(slf_recon, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)
            # test SLF image MAE
            slf_mae1 += F.l1_loss(slf_coarse, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)
            slf_mae2 += F.l1_loss(slf_recon, target_slf, reduction='sum').item() / (self.args.K0 * self.args.K1)

        slf_rmse1 /= len(test_loader.dataset)
        slf_rmse1 = np.sqrt(slf_rmse1)
        slf_rmse2 /= len(test_loader.dataset)
        slf_rmse2 = np.sqrt(slf_rmse2)
        slf_mae1 /= len(test_loader.dataset)
        slf_mae2 /= len(test_loader.dataset)
        print('====> Test Model:')
        print('====> Noise Level: ' + noise_level)
        print('====> First stage SLF rmse: {:.6f}'.format(slf_rmse1))
        print('====> Refined SLF rmse: {:.6f}'.format(slf_rmse2))
        print('====> First stage SLF mae: {:.6f}'.format(slf_mae1))
        print('====> Refined SLF mae: {:.6f}'.format(slf_mae2))