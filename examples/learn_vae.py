"""
This module implements the VAE training for DW-MRI

Authors:
    Julius Glaser <julius.glaser@fau.de>
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

from __future__ import division
import argparse
import copy
import h5py
import os
import pathlib
import torch
import yaml

import numpy as np
import torch.utils.data as data

from datetime import datetime

from deepdwi.models import bloch, autoencoder
from deepdwi.recons import linsub

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
print('> HOME_DIR: ', HOME_DIR)

DATA_DIR = DIR.rsplit('/', 1)[0] + '/data'
print('> DATA_DIR: ', DATA_DIR)

# %%
class network_parameters:
    def __init__(self, config):
        self.dvs = config['dvs']

        self.model = config['model']
        self.device = config['device']
        self.latent = config['latent']
        self.epochs = config['epochs']
        self.batch_size_train = config['batch_size_train']
        self.batch_size_test = config['batch_size_test']

        self.noise_range = config['noise_range']
        self.learning_rate = config['learning_rate']
        self.D_loss_weight = config['D_loss_weight']

        self.kld_start_epoch = config['kld_start_epoch']
        self.kld_restart = config['kld_restart']
        self.kld_max_weight = config['kld_max_weight']
        self.kld_weight_increase = config['kld_weight_increase']

        self.optimizer = config['optimizer']
        self.loss_function = config['loss_function']
        self.device = config['device']
        self.test_epoch_step = config['test_epoch_step']

        self.kld_weight = 0
        self.tested_once = False
        self.config = config

        self.N_diff = 0

    def set_output_dir(self, base_dir):

        data_str = self.dvs.rsplit('/', 1)[1]
        data_str = data_str.rsplit('_dvs.h5', 1)[0]

        now = datetime.now()
        dir_name = now.strftime("%Y-%m-%d")

        dir_name += '_' + self.model

        dir_name += '_' + data_str
        dir_name += '_latent-' + str(self.latent).zfill(2)

        self.acquisition_dir = os.path.join(base_dir, dir_name)
        return self.acquisition_dir

    def set_model(self, inputFeatures, device):
        if self.model == 'DAE':
            return autoencoder.DAE(input_features=inputFeatures, latent_features=self.latent).to(device)
        elif self.model == 'VAE':
            return autoencoder.VAE(input_features=inputFeatures, latent_features=self.latent).to(device)

    def set_lossf(self):
        if self.loss_function == 'MSE':
            return torch.nn.MSELoss()
        elif self.loss_function == 'L1':
            return torch.nn.L1Loss()
        elif self.loss_function == 'Huber':
            return torch.nn.HuberLoss()

    def set_optim(self, model):
        if self.optimizer == 'SGD':
            return torch.optim.SGD(model.parameters(), self.learning_rate, weight_decay=1E-5)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(model.parameters(), self.learning_rate)

    def set_device(self):
        return torch.device(self.device)

    def update_kld_weight(self, training_epoch):
            if training_epoch >= self.kld_start_epoch:
                if self.kld_weight < self.kld_max_weight:
                    self.kld_weight += self.kld_weight_increase
                elif self.kld_restart:
                    if training_epoch >= self.kld_start_epoch + self.kld_max_weight/self.kld_weight_increase + 1:
                        self.kld_weight = 0

    def __str__(self) -> str:
        output_str = ''
        for entry in self.config:
            if (entry.startswith('kld')) and self.model == 'DAE':
                pass
            else:
                output_str += f'> used {entry}: {self.config[entry]}\n'
        return output_str

class losses:
    def __init__(self, model: str):
        if model == 'DAE':
            self.train = []
            self.test = []
            self.mse = []

            self.D = []
            self.D_test = []

            self.D_recon_over_epochs = []
            self.D_noisy_over_epochs = []
            self.D_clean_over_epochs = []

        elif model == 'VAE':
            self.train = []
            self.test = []
            self.mse = []

            self.kld = []
            self.testKld = []
            self.recon = []
            self.testRecon = []

            self.D = []
            self.D_test = []

            self.standards = []
            self.means = []

            self.D_recon_over_epochs = []
            self.D_noisy_over_epochs = []
            self.D_clean_over_epochs = []

    def create_loss_file(self, network_parameters: network_parameters):
        epoch = network_parameters.epochs
        latent = network_parameters.latent
        model = network_parameters.model

        lossFile = h5py.File(network_parameters.acquisition_dir + '/valid_' + model + '_Latent' + str(latent).zfill(2) + 'EpochTrain' + str(epoch) + 'Loss.h5', 'w')

        lossFile.create_dataset('testLoss' + str(epoch), data=torch.tensor(self.test).detach().cpu().numpy())
        lossFile.create_dataset('trainLoss' + str(epoch), data=torch.tensor(self.train).detach().cpu().numpy())
        lossFile.create_dataset('mseLoss' + str(epoch), data=torch.tensor(self.mse).detach().cpu().numpy())

        lossFile.create_dataset('D_recon' + str(epoch), data=torch.tensor(self.D_recon_over_epochs).detach().cpu().numpy())
        lossFile.create_dataset('D_noisy' + str(epoch), data=torch.tensor(self.D_noisy_over_epochs).detach().cpu().numpy())
        lossFile.create_dataset('D_clean' + str(epoch), data=torch.tensor(self.D_clean_over_epochs).detach().cpu().numpy())

        if model == 'VAE':
            lossFile.create_dataset('kldLoss' + str(epoch), data=torch.tensor(self.kld).detach().cpu().numpy())
            lossFile.create_dataset('reconLoss' + str(epoch), data=torch.tensor(self.recon).detach().cpu().numpy())
            lossFile.create_dataset('testReconLoss' + str(epoch), data=torch.tensor(self.testRecon).detach().cpu().numpy())
            lossFile.create_dataset('testKldLoss' + str(epoch), data=torch.tensor(self.testKld).detach().cpu().numpy())
            lossFile.create_dataset('means' + str(epoch), data=torch.tensor(self.means).detach().cpu().numpy())
            lossFile.create_dataset('standards' + str(epoch), data=torch.tensor(self.standards).detach().cpu().numpy())

        lossFile.close()

    def create_mse_loss_txt_file(self, network_parameters: network_parameters, model):
        #Create txt config file:
        completeName = os.path.join(network_parameters.acquisition_dir, "mseLoss.txt")

        txtFile = open(completeName, "w")

        txtFile.write("\n\n\n Results in Testing:\n\n")
        for line in range(len(self.mse)):
            txtFile.write("Loss with MSE after Epoch {} : {:12.7f}\n".format(network_parameters.test_epoch_step*line, self.mse[line]))
        txtFile.write('\n\n')
        txtFile.write(model.__str__())
        txtFile.close()

# %%
def train(network_parameters, loader_train, optim, model, device, loss_function, epoch, Losses):
    model.train()

    train_loss = 0.0
    kld_loss = 0.0
    recon_loss = 0.0
    train_loss = 0.0

    for batch_idx, (noisy_t, clean_t, _) in enumerate(loader_train):

        noisy_t = noisy_t.type(torch.FloatTensor).to(device)
        clean_t = clean_t.type(torch.FloatTensor).to(device)

        optim.zero_grad()

        if network_parameters.model == 'DAE':
            recon_t = model(noisy_t)
            loss = loss_function(recon_t, clean_t)
        elif network_parameters.model == 'VAE':
            recon_t, mu, logvar = model(noisy_t)

            loss_lossf = loss_function(recon_t, clean_t)
            loss_lossf *= 32
            KLD = autoencoder.loss_function_kld(mu, logvar)
            loss = loss_lossf + network_parameters.kld_weight*KLD
            kld_loss += (network_parameters.kld_weight*KLD).item()
            recon_loss += loss_lossf.item()

        loss.backward()
        optim.step()
        train_loss += loss.item()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {:4d} [{:10d}/{:10d} ({:3.0f}%)]\tLoss: {:12.6f}'.format(
                epoch, batch_idx * len(noisy_t), len(loader_train.dataset),
                100. * batch_idx / len(loader_train),
                loss.item() / len(noisy_t)))

    Losses.train.append(train_loss / len(loader_train.dataset))
    #Losses.D.append(D_loss / len(loader_train.dataset))

    if network_parameters.model == 'VAE':
        Losses.kld.append(kld_loss / len(loader_train.dataset))
        Losses.recon.append(recon_loss / len(loader_train.dataset))

    print('====> Epoch: {} Average loss Training: {:12.6f}'.format(epoch, Losses.train[-1]))

    return Losses

# %%
def test(network_parameters, loader_test, model, device, loss_function, epoch, h5pyFile, Losses):
    model.eval()
    kld_loss = 0.0
    recon_loss = 0.0
    mse_loss = 0.0
    test_loss = 0.0

    test_clean = []
    test_noisy = []
    test_noise_amount = []
    test_recon = []
    D_clean_values = []

    with torch.no_grad():
        for batch_idx, (noisy_t, clean_t, noise) in enumerate(loader_test):

            noisy_t = noisy_t.type(torch.FloatTensor).to(device)
            clean_t = clean_t.type(torch.FloatTensor).to(device)

            if network_parameters.model == 'DAE':
                recon_t = model(noisy_t)
                loss = loss_function(recon_t, clean_t)
            elif network_parameters.model == 'VAE':
                recon_t, mu, logvar = model(noisy_t)


                loss_lossf = loss_function(recon_t, clean_t)
                KLD = autoencoder.loss_function_kld(mu, logvar)

                kld_loss += (network_parameters.kld_weight*KLD).item()
                recon_loss += loss_lossf.item()
                loss = loss_lossf*32 + network_parameters.kld_weight*KLD

                #if D_loss_weight > 1:
                #    loss = loss_lossf + weight*KLD + D_loss_weight * loss_D
                #else:
                #    pass

            test_recon.append(recon_t.cpu().detach().numpy())
            if not network_parameters.tested_once:
                test_noisy.append(noisy_t.cpu().detach().numpy())
                test_clean.append(clean_t.cpu().detach().numpy())
                test_noise_amount.append(noise.cpu().detach().numpy())
                # D_clean_values.append(D_clean.cpu().detach().numpy())

            '''if last_test:
                D_recon = epi.get_D(B, np.transpose(recon_t),  fit_only_tensor=True)
                D_recon_arr.append(D_recon)
                D_noisy = epi.get_D(B, np.transpose(noisy_t),  fit_only_tensor=True)
                #D_clean = epi.get_D(B, np.transpose(clean_t),  fit_only_tensor=True)
                #D_clean_arr.append(D_clean)
                D_noisy_arr.append(D_noisy)'''


            loss_mse = autoencoder.loss_function_mse(recon_t, clean_t)
            mse_loss += loss_mse
            test_loss += loss

    h5pyFile.create_dataset('reconEpoch' + str(epoch), data=np.array(test_recon))
    if not network_parameters.tested_once:
        h5pyFile.create_dataset('noisyEpoch' + str(epoch), data=np.array(test_noisy))
        h5pyFile.create_dataset('cleanEpoch' + str(epoch), data=np.array(test_clean))
        h5pyFile.create_dataset('noiseParam' + str(epoch), data=np.array(test_noise_amount))
        h5pyFile.create_dataset('D_values' + str(epoch), data=np.array(D_clean_values))
        network_parameters.tested_once = True

    Losses.test.append(test_loss / len(loader_test.dataset))
    Losses.mse.append(mse_loss / len(loader_test.dataset))

    if network_parameters.model == 'VAE':
        Losses.testKld.append(kld_loss / len(loader_test.dataset))
        Losses.testRecon.append(recon_loss / len(loader_test.dataset))
        Losses.means.append(mu.detach().cpu().numpy())
        std = torch.exp(0.5 * logvar)
        Losses.standards.append(std.detach().cpu().numpy())

    print('====> Epoch: {} Average loss Testing: {:12.7f}'.format(epoch, Losses.test[-1]))
    print('====> Epoch: {} Average mse_loss Testing: {:12.7f}'.format(epoch, Losses.mse[-1]))

    #save trained model
    out_str=''
    if epoch == network_parameters.epochs:
        out_str = '/train_' + network_parameters.model + '_Latent' + str(network_parameters.latent).zfill(2) + '_final.pt'
    else:
        out_str = '/train_' + network_parameters.model + '_Latent' + str(network_parameters.latent).zfill(2) + '_epoch' + str(epoch).zfill(3) + '.pt'
    torch.save(model.state_dict(), network_parameters.acquisition_dir + out_str)

    return Losses

# %%
def setup(config_dict):

    NetworkParameters = network_parameters(config_dict)
    print(NetworkParameters)

    # read in diffusion vector set (dvs) file
    f = h5py.File(HOME_DIR + config_dict['dvs'], 'r')
    bvals = f['bvals'][:]
    bvecs = f['bvecs'][:]
    f.close()

    NetworkParameters.N_diff = len(bvals)

    if bvals.ndim == 1:
        bvals = bvals[:, np.newaxis]

    x_clean, original_D = bloch.model_DTI(bvals, bvecs)

    B = bloch.get_B(bvals, bvecs)

    return NetworkParameters, x_clean, original_D, B

# %%
class DwiDataset(data.Dataset):

    def __init__(self, x_noisy, x_clean, noise_amount, transform=None):

        self.x_noisy = x_noisy
        self.x_clean = x_clean
        self.noise_amount = noise_amount

        print('> DwiDataset x_noisy shape: ', x_noisy.shape)

        self.N_atom = x_clean.shape[0]
        self.N_diff = x_clean.shape[1]


        # transforms.ToTensor() scales images!!!
        # if transform is None:
        #     transform = transforms.Compose([transforms.ToTensor()])

        self.transform = transform

    def __len__(self):

        assert (len(self.x_noisy) == len(self.x_clean))
        return len(self.x_noisy)

    def __getitem__(self, idx):

        x_noisy = self.x_noisy[idx]
        x_clean = self.x_clean[idx]
        noise_amount = self.noise_amount[idx]

        if self.transform is not None:
            x_noisy = self.transform(x_noisy)
            x_clean = self.transform(x_clean)

        return (x_noisy, x_clean, noise_amount)

# %%
def add_noise(x_clean, scale):

    x_noisy = x_clean + np.random.normal(loc = 0,
                                         scale = scale,
                                         size=x_clean.shape)

    x_noisy[x_noisy < 0.] = 0.
    x_noisy[x_noisy > 1.] = 1.

    return x_noisy

# %%
def create_noisy_dataset(train_set, test_set, NetworkParameters):

    train_set_full = copy.deepcopy(train_set)
    test_set_full = copy.deepcopy(test_set)


    for id in range(1, NetworkParameters.noise_range):
        train_set_copy = copy.deepcopy(train_set)
        test_set_copy = copy.deepcopy(test_set)

        sd = 0.01 + id * 0.03


        train_set_copy[:][0][:] = add_noise(train_set_copy[:][1], sd)
        train_set_copy[:][2][:] = id

        test_set_copy[:][0][:] = add_noise(test_set_copy[:][1], sd)
        test_set_copy[:][2][:] = id

        # TODO: remove for loops
        # for index in range(len(train_set_copy)):
        #     train_set_copy[index][0][:] = add_noise(train_set_copy[index][1], sd)
        #     train_set_copy[index][2][:] = id

        # for index in range(len(test_set_copy)):
        #     test_set_copy[index][0][:] = add_noise(test_set_copy[index][1], sd)
        #     test_set_copy[index][2][:] = id

        train_set_full = data.ConcatDataset([train_set_full, train_set_copy])
        test_set_full =  data.ConcatDataset([test_set_full, test_set_copy])

    print('train set size ', len(train_set))
    print('test set size ', len(test_set))

    return train_set_full, test_set_full

def create_data_loader(x_clean, original_D, NetworkParameters):

    noise_amount = np.zeros(shape=(1, x_clean.shape[1]))
    noise_amount = np.transpose(noise_amount)

    x_clean = np.transpose(x_clean)
    original_D = np.transpose(original_D)

    dataset = DwiDataset(x_clean, x_clean, noise_amount)

    datalen = len(dataset)

    train_siz = int(datalen * 0.9)
    test_siz = datalen - train_siz

    train_set, test_set = data.random_split(dataset, [train_siz, test_siz])

    train_set_noised, test_set_noised = create_noisy_dataset(train_set, test_set, NetworkParameters)

    loader_train = data.DataLoader(train_set_noised, batch_size=NetworkParameters.batch_size_train, shuffle=True)

    loader_test  = data.DataLoader(test_set_noised, batch_size=NetworkParameters.batch_size_test, shuffle=False)

    return loader_train, loader_test


# %%
def main():

    torch.manual_seed(0)

    # %% argument parser
    parser = argparse.ArgumentParser(description='learn VAE.')

    parser.add_argument('--config', type=str,
                        default='/configs/learn_vae.yaml',
                        help='yaml config file for zsssl')

    args = parser.parse_args()

    # %%
    with open(HOME_DIR + args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    NetworkParameters, x_clean, original_D, B = setup(config_dict)

    loader_train, loader_test = create_data_loader(x_clean, original_D, NetworkParameters)

    device = NetworkParameters.set_device()
    model = NetworkParameters.set_model(NetworkParameters.N_diff, device)
    loss_function = NetworkParameters.set_lossf()
    optimizer = NetworkParameters.set_optim(model)

    RECON_DIR = NetworkParameters.set_output_dir(DIR)

    print('> RECON_DIR: ', RECON_DIR)
    # make a new directory if not exist
    pathlib.Path(RECON_DIR).mkdir(parents=True, exist_ok=True)

    f = h5py.File(RECON_DIR + '/valid_epoch-' + str(NetworkParameters.epochs).zfill(3) + '.h5', 'w')

    Losses = losses(NetworkParameters.model)

    # %% training
    for epoch in range(1, NetworkParameters.epochs+1, 1):

        Losses = train(NetworkParameters, loader_train, optimizer, model, device, loss_function, epoch, Losses)

        if epoch == NetworkParameters.epochs or epoch % NetworkParameters.test_epoch_step == 0:
            Losses = test(NetworkParameters, loader_test, model, device, loss_function, epoch, f, Losses)

        NetworkParameters.update_kld_weight(epoch)

    Losses.create_loss_file(NetworkParameters)
    Losses.create_mse_loss_txt_file(NetworkParameters, model)

    f.close()

if __name__ == "__main__":
    main()