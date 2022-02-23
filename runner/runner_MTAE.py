import numpy as np
import os
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.utils import save_image
from model.MTAE import MTAE
import random
import scipy.io as sio
from torchinfo import summary


# runner for Multi-task AE
class runner_MTAE():
    def __init__(self, args):
        self.args = args
        cuda_flag = not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_flag else "cpu")
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        # torch.use_deterministic_algorithms(True)
        self.path = self.args.run + 'MTAE/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def get_optimizer(self, parameters):
        if self.args.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.999), eps=1e-7)
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
        train_input_RSS = training_data['RSS']
        RSS_min = RSS_minmax['RSS_min']
        RSS_max = RSS_minmax['RSS_max']
        train_input_RSS = (train_input_RSS - RSS_min) / (RSS_max - RSS_min)
        train_input_RSS = train_input_RSS.reshape(-1, self.args.N, self.args.P, self.args.P)
        # normalized clean RSS measurement input (as target for decoder) (shape: [num_sample, NxPxP])
        train_denoise_RSS = training_data['RSS_denoise']
        RSS_denoise_min = RSS_minmax['RSS_denoise_min']
        RSS_denoise_max = RSS_minmax['RSS_denoise_max']
        train_denoise_RSS = (train_denoise_RSS - RSS_denoise_min) / (RSS_denoise_max - RSS_denoise_min)
        train_denoise_RSS = train_denoise_RSS.reshape(-1, self.args.N, self.args.P, self.args.P)
        # SLF image (shape: [num_sample, K[0]*K[1]])
        train_slf_img = training_data['slf_img']
        train_slf_img = train_slf_img.reshape(-1, self.args.K0 * self.args.K1)
        # b: bias. alpha: path loss exponent. ab_norm: [b_norm, alpha_norm] (shape: [num_sample, N+1])
        train_ab = training_data['ab_norm']
        # noise level classes (shape: [num_sample, 1]) (classes: 0, 1, 2)
        train_noise_level = training_data['sig_epsilon_class']
        train_noise_level = np.squeeze(train_noise_level)

        train_input_RSS = torch.from_numpy(train_input_RSS).float()
        train_denoise_RSS = torch.from_numpy(train_denoise_RSS).float()
        train_slf_img = torch.from_numpy(train_slf_img).float()
        train_ab = torch.from_numpy(train_ab).float()
        train_noise_level = torch.from_numpy(train_noise_level).long()

        dataset = TensorDataset(train_input_RSS, train_denoise_RSS, train_slf_img, train_ab, train_noise_level)

        train_size = int(len(dataset)*0.9)
        val_size = int(len(dataset)*0.1)
        dataset_train, dataset_val = random_split(dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(self.args.seed))

        # Load testing data
        testing_data = sio.loadmat('./data/testing_data.mat')
        # noise level classes (shape: [num_sample, 1]) (classes: 0, 1, 2)
        test_noise_class = testing_data['sig_epsilon_class']
        test_noise_class = np.squeeze(test_noise_class)
        idx = (test_noise_class == 0) | (test_noise_class == 1) | (test_noise_class == 2)
        if noise_level == 'all':
            pass
        elif noise_level == 'low':
            idx = (test_noise_class == 0)
        elif noise_level == 'mid':
            idx = (test_noise_class == 1)
        elif noise_level == 'high':
            idx = (test_noise_class == 2)
        else:
            raise NotImplementedError('Noise level {} not understood.'.format(noise_level))
        test_noise_class = test_noise_class[idx]
        # normalized noisy RSS measurement input (shape: [num_sample, NxPxP])
        test_input_RSS = testing_data['RSS'][idx]
        test_input_RSS = (test_input_RSS - RSS_min) / (RSS_max - RSS_min)
        test_input_RSS = test_input_RSS.reshape(-1, self.args.N, self.args.P, self.args.P)

        # normalized clean RSS measurement input (as target for decoder) (shape: [num_sample, NxPxP])
        test_denoise_RSS = testing_data['RSS_denoise'][idx]
        test_denoise_RSS = (test_denoise_RSS - RSS_denoise_min) / (RSS_denoise_max - RSS_denoise_min)
        test_denoise_RSS = test_denoise_RSS.reshape(-1, self.args.N, self.args.P, self.args.P)

        # SLF image (shape: [1, K[0]*K[1]])
        test_slf_img = testing_data['slf_img'][idx]
        test_slf_img = test_slf_img.reshape(-1, self.args.K0 * self.args.K1)

        # b: bias. alpha: path loss exponent. ab_norm: [b_norm, alpha_norm] (shape: [num_sample, N+1])
        test_ab = testing_data['ab_norm'][idx]

        test_input_RSS = torch.from_numpy(test_input_RSS).float()
        test_denoise_RSS = torch.from_numpy(test_denoise_RSS).float()
        test_slf_img = torch.from_numpy(test_slf_img).float()
        test_ab = torch.from_numpy(test_ab).float()
        test_noise_class = torch.from_numpy(test_noise_class).long()

        dataset_test = TensorDataset(test_input_RSS, test_denoise_RSS, test_slf_img, test_ab, test_noise_class)

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
        train_loss = 0                  # total loss
        train_loss_task0 = 0            # loss for reconstruct RSS
        train_loss_task1 = 0            # loss for SLF image estimation
        train_loss_task2 = 0            # loss for parameters estimation
        train_loss_task3 = 0            # loss for noise level prediction
        train_acc = 0                   # accuracy for noise level prediction
        for batch_idx,  (input_RSS, target_RSS, target_SLF, target_ab, target_noise_level) in enumerate(train_loader):
            input_RSS = input_RSS.to(self.device)
            target_RSS = target_RSS.to(self.device)
            target_SLF = target_SLF.to(self.device)
            target_ab = target_ab.to(self.device)
            target_noise_level = target_noise_level.to(self.device)
            optimizer.zero_grad()
            RSS, SLF, ab, noise_level = model(input_RSS)
            loss_task0 = F.binary_cross_entropy(RSS, target_RSS, reduction='sum')/(self.args.N * self.args.P * self.args.P)
            loss_task1 = F.binary_cross_entropy(SLF, target_SLF, reduction='sum')/(self.args.K0 * self.args.K1)
            loss_task2 = F.binary_cross_entropy(ab, target_ab, reduction='sum')/(self.args.N + 1)
            loss_task3 = F.cross_entropy(noise_level, target_noise_level, reduction='sum')
            loss = self.args.lambda1_MTAE * loss_task0 + self.args.lambda2_MTAE * loss_task1 \
                   + self.args.lambda3_MTAE * loss_task2 + self.args.lambda4_MTAE * loss_task3
            train_loss += loss.item()
            train_loss_task0 += loss_task0.item()
            train_loss_task1 += loss_task1.item()
            train_loss_task2 += loss_task2.item()
            train_loss_task3 += loss_task3.item()
            _, noise_pred = torch.max(noise_level, 1)
            acc_task3 = (noise_pred == target_noise_level).sum()
            train_acc += acc_task3.item()
            loss /= input_RSS.size(0)
            loss.backward()
            optimizer.step()

            if batch_idx % self.args.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss0: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}\tAcc: {:.6f}'.format(
                        epoch, batch_idx * len(input_RSS), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                               loss.item(),
                               loss_task0.item() / len(input_RSS), loss_task1.item() / len(input_RSS),
                               loss_task2.item() / len(input_RSS), loss_task3.item() / len(input_RSS), acc_task3.item() / len(input_RSS)))

        train_loss /= len(train_loader.dataset)
        train_loss_task0 /= len(train_loader.dataset)
        train_loss_task1 /= len(train_loader.dataset)
        train_loss_task2 /= len(train_loader.dataset)
        train_loss_task3 /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss))
        print('====> RSS loss: {:.6f}'.format(train_loss_task0))
        print('====> SLF loss: {:.6f}'.format(train_loss_task1))
        print('====> ab loss: {:.6f}'.format(train_loss_task2))
        print('====> noise level loss: {:.6f}'.format(train_loss_task3))
        print('====> noise level Accuracy: {:.4f}'.format(train_acc))
        return train_loss, train_loss_task0, train_loss_task1, train_loss_task2, train_loss_task3, train_acc

    def validate(self, model, val_loader):
        model.eval()
        val_loss = 0                  # total loss
        val_bce_slf = 0
        val_rmse_slf = 0              # slf reconstruction loss
        val_mae_slf = 0              # slf reconstruction loss
        for batch_idx,  (input_RSS, target_RSS, target_SLF, target_ab, target_noise_level) in enumerate(val_loader):
            input_RSS = input_RSS.to(self.device)
            target_RSS = target_RSS.to(self.device)
            target_SLF = target_SLF.to(self.device)
            target_ab = target_ab.to(self.device)
            target_noise_level = target_noise_level.to(self.device)
            RSS, SLF, ab, noise_level = model(input_RSS)
            loss_task0 = F.binary_cross_entropy(RSS, target_RSS, reduction='sum')/(self.args.N * self.args.P * self.args.P)
            loss_task1 = F.binary_cross_entropy(SLF, target_SLF, reduction='sum')/(self.args.K0 * self.args.K1)
            loss_task2 = F.binary_cross_entropy(ab, target_ab, reduction='sum')/(self.args.N + 1)
            loss_task3 = F.cross_entropy(noise_level, target_noise_level, reduction='sum')
            loss = self.args.lambda1_MTAE * loss_task0 + self.args.lambda2_MTAE * loss_task1 \
                   + self.args.lambda3_MTAE * loss_task2 + self.args.lambda4_MTAE * loss_task3
            val_loss += loss.item()
            val_bce_slf += loss_task1.item()
            slf_mse = F.mse_loss(SLF, target_SLF, reduction='sum')/(self.args.K0 * self.args.K1)
            slf_mae = F.l1_loss(SLF, target_SLF, reduction='sum')/(self.args.K0 * self.args.K1)
            val_rmse_slf += slf_mse.item()
            val_mae_slf += slf_mae.item()
        val_loss /= len(val_loader.dataset)
        val_bce_slf /= len(val_loader.dataset)
        val_rmse_slf /= len(val_loader.dataset)
        val_rmse_slf = np.sqrt(val_rmse_slf)
        val_mae_slf /= len(val_loader.dataset)
        print('====> Validation set loss: {:.6f}'.format(val_loss))
        print('====> Validation set SLF rmse: {:.6f}'.format(val_rmse_slf))
        print('====> Validation set SLF mae: {:.6f}'.format(val_mae_slf))
        return val_loss, val_bce_slf, val_rmse_slf, val_mae_slf

    def test(self, model, test_loader):
        model.eval()
        test_loss = 0  # total loss
        test_rmse = 0  # rmse
        test_loss_task0 = 0  # loss for reconstruct RSS
        test_loss_task1 = 0  # loss for SLF image estimation
        test_loss_task2 = 0  # loss for parameters estimation
        test_loss_task3 = 0  # loss for noise level prediction
        with torch.no_grad():
            for batch_idx, (input_RSS, target_RSS, target_SLF, target_ab, target_noise_level) in enumerate(test_loader):
                input_RSS = input_RSS.to(self.device)
                target_RSS = target_RSS.to(self.device)
                target_SLF = target_SLF.to(self.device)
                target_ab = target_ab.to(self.device)
                target_noise_level = target_noise_level.to(self.device)
                RSS, SLF, ab, noise_level = model(input_RSS)
                loss_task0 = F.binary_cross_entropy(RSS, target_RSS, reduction='sum') / (
                            self.args.N * self.args.P * self.args.P)
                loss_task1 = F.binary_cross_entropy(SLF, target_SLF, reduction='sum') / (self.args.K0 * self.args.K1)
                loss_task2 = F.binary_cross_entropy(ab, target_ab, reduction='sum') / (self.args.N + 1)
                loss_task3 = F.cross_entropy(noise_level, target_noise_level, reduction='sum')
                loss = self.args.lambda1_MTAE * loss_task0 + self.args.lambda2_MTAE * loss_task1 \
                       + self.args.lambda3_MTAE * loss_task2 + self.args.lambda4_MTAE * loss_task3

                rmse = F.mse_loss(SLF, target_SLF, reduction='sum') / (self.args.K0 * self.args.K1)

                test_loss += loss.item()
                test_rmse += rmse.item()
                test_loss_task0 += loss_task0.item()
                test_loss_task1 += loss_task1.item()
                test_loss_task2 += loss_task2.item()
                test_loss_task3 += loss_task3.item()

        test_loss /= len(test_loader.dataset)
        test_loss_task0 /= len(test_loader.dataset)
        test_loss_task1 /= len(test_loader.dataset)
        test_loss_task2 /= len(test_loader.dataset)
        test_loss_task3 /= len(test_loader.dataset)
        test_acc = self.test_accuracy(model, test_loader)
        test_rmse /= len(test_loader.dataset)
        test_rmse = np.sqrt(test_rmse)
        print('====> Test set loss: {:.6f}'.format(test_loss))
        print('====> Test set SLF RMSE: {:.6f}'.format(test_rmse))
        print('====> Test set RSS loss: {:.6f}'.format(test_loss_task0))
        print('====> Test set SLF loss: {:.6f}'.format(test_loss_task1))
        print('====> Test set ab loss: {:.6f}'.format(test_loss_task2))
        print('====> Test set noise level loss: {:.6f}'.format(test_loss_task3))
        print('====> Test set noise level Accuracy: {:.4f}'.format(test_acc))
        return test_loss, test_loss_task0, test_loss_task1, test_loss_task2, test_loss_task3, test_acc

    def train_save(self):
        model = MTAE(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))

        def weights_init(m):
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

        model.apply(weights_init)
        model = model.to(self.device)
        summary(model, input_size=(self.args.batch_size, self.args.N, self.args.P, self.args.P))
        train_loader, val_loader, test_loader = self.load_data()
        optimizer = self.get_optimizer(model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=8, min_lr=1e-5)
        train_set = []
        val_set = []
        test_set = []       # for test set
        train_loss_path = self.path + 'train_loss_' + str(self.args.n_epochs_MTAE) + '.npy'
        val_loss_path = self.path + 'val_loss_' + str(self.args.n_epochs_MTAE) + '.npy'
        test_loss_path = self.path + 'test_loss_' + str(self.args.n_epochs_MTAE) + '.npy'
        model_path = self.path + 'MTAE_' + str(self.args.n_epochs_MTAE) + '.pth'
        current_val_slf_loss = np.inf
        for epoch in range(1, self.args.n_epochs_MTAE+1):
            print("Epoch %d learning rate：%f" % (epoch, optimizer.param_groups[0]['lr']))
            train_loss, train_loss_task0, train_loss_task1, train_loss_task2, train_loss_task3, train_acc = self.train(model,
                                                                                                            train_loader,
                                                                                                            optimizer,
                                                                                                            epoch)
            train_set.append([train_loss, train_loss_task0, train_loss_task1, train_loss_task2, train_loss_task3, train_acc])
            print('====> Validation Loss for Epoch {:d}'.format(epoch))
            val_loss, val_bce_slf, val_slf_rmse, val_slf_mae = self.validate(model, val_loader)
            val_set.append([val_loss, val_bce_slf, val_slf_rmse, val_slf_mae])
            scheduler.step(val_loss)
            print('====> Test for Epoch {:d}'.format(epoch))
            test_loss, test_loss_task0, test_loss_task1, test_loss_task2, test_loss_task3, test_acc = self.test(model,
                                                                                                                test_loader)
            test_set.append([test_loss, test_loss_task0, test_loss_task1, test_loss_task2, test_loss_task3, test_acc])
            if epoch % self.args.save_freq == 0:
                ckpt_path = self.path + 'model_' + str(epoch) + '.pth'
                torch.save(model.state_dict(), ckpt_path)
                print('checkpoint{}.pth saved!'.format(epoch))

            if val_slf_rmse < current_val_slf_loss:
                torch.save(model.state_dict(), model_path)
                current_val_slf_loss = val_slf_rmse

        train_loss = np.asarray(train_set).reshape(-1, 6)
        val_loss = np.asarray(val_set).reshape(-1, 4)
        test_loss = np.asarray(test_set).reshape(-1, 6)
        np.save(train_loss_path, train_loss)
        np.save(val_loss_path, val_loss)
        np.save(test_loss_path, test_loss)

    def test_accuracy(self, model, test_loader):
        model.eval()
        acc = 0
        for i, (input_RSS, *_, target_noise_level) in enumerate(test_loader):
            input_RSS = input_RSS.to(self.device)
            *_, noise_pred = model(input_RSS)  # noise_pred: [batch_size, 3]
            noise_pred = noise_pred.detach().cpu().numpy()
            noise_pred = np.argmax(noise_pred, axis=1)
            labels = target_noise_level.detach().cpu().numpy()
            acc += np.sum((noise_pred == labels).astype(int))
        acc /= len(test_loader.dataset)
        return acc

    def test_model(self, noise_level='all'):
        model = MTAE(M=self.args.M, P=self.args.P, K=(self.args.K0, self.args.K1))
        model = model.to(self.device)
        model_path = self.path + 'MTAE_' + str(self.args.n_epochs_MTAE) + '.pth'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        *_, test_loader = self.load_data(noise_level)
        noise_class = 0  # noise_class = 0 stands for all noise classes
        if noise_level == 'low':
            noise_class = 1
        elif noise_level == 'mid':
            noise_class = 2
        elif noise_level == 'high':
            noise_class = 3

        # test model for testing set
        slf_rmse = 0
        slf_mae = 0
        ab_rmse = 0
        for batch_idx, (input_RSS, target_RSS, target_SLF, target_ab, target_noise_level) in enumerate(
                test_loader):
            input_RSS = input_RSS.to(self.device)
            target_RSS = target_RSS.to(self.device)
            target_SLF = target_SLF.to(self.device)
            target_ab = target_ab.to(self.device)
            target_noise_level = target_noise_level.to(self.device)
            RSS, SLF, ab, noise_pred = model(input_RSS)
            # plot SLF reconstruction image for image1
            n = min(target_RSS.size(0), 8)
            comparison = torch.cat([target_SLF.view(target_RSS.size(0), 1, self.args.K0, self.args.K1)[:n],
                                    SLF.view(SLF.size(0), 1, self.args.K0, self.args.K1)[:n]])
            save_image(comparison.cpu(),
                       self.path + 'reconstruction_SLF_noise' + str(noise_class) + '_' + str(batch_idx) + '.png',
                       nrow=n)
            # test SLF image RMSE
            slf_rmse += F.mse_loss(SLF, target_SLF, reduction='sum').item() / (self.args.K0 * self.args.K1)
            # test SLF image MAE
            slf_mae += F.l1_loss(SLF, target_SLF, reduction='sum').item() / (self.args.K0 * self.args.K1)
            # test ab parameters RMSE
            ab_rmse += F.mse_loss(ab, target_ab, reduction='sum').item() / (self.args.N + 1)

        slf_rmse /= len(test_loader.dataset)
        slf_rmse = np.sqrt(slf_rmse)
        slf_mae /= len(test_loader.dataset)
        ab_rmse /= len(test_loader.dataset)
        ab_rmse = np.sqrt(ab_rmse)
        noise_acc = self.test_accuracy(model, test_loader)
        print('====> Test for Testing Set:')
        print('====> Noise_level: ' + noise_level)
        print('====> Test set SLF rmse: {:.6f}'.format(slf_rmse))
        print('====> Test set SLF mae: {:.6f}'.format(slf_mae))
        print('====> Test set ab rmse: {:.6f}'.format(ab_rmse))
        print('====> Test set noise level Accuracy: {:.4f}'.format(noise_acc))