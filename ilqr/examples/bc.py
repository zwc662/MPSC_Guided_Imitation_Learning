import torch.nn as nn
import torch.nn.functional as F
import torch

import argparse
from pathlib import PurePath as Path
import scipy.optimize
import numpy as np

from torch.autograd import Variable
import torch.utils.data as data_utils

import logging
import time
import os

import pickle



class mlp(nn.Module):
    def __init__(self, input_size, output_size):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_size, 2 * input_size)
        self.fc2 = nn.Linear(2 * input_size, 4 * input_size)
        self.fc3 = nn.Linear(4 * input_size, 3 * input_size)
        self.fc4 = nn.Linear(3 * input_size, output_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x_0 = nn.functional.sigmoid(x[:, 0]).unsqueeze(1)
        x_1 = nn.functional.tanh(x[:, 1]).unsqueeze(1)
        y = torch.cat((x_0, x_1), dim = 1)
        return y


class NeuralNetwork:
    def __init__(self, input_size = 4, output_size = 2, batch_size = 10, model_name = 'mlp', checkpoint = None):
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size

        self.model_name = model_name

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = mlp(self.input_size, self.output_size).to(self.device)
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location = self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum = 0.9)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.8, 0.999), eps=1e-05, weight_decay=0.)
        self.dataset = None

    def data_process(self, X = None, Y = None, paths = None):
        if paths is not None:
            X = []
            Y = []
            for path in paths:
                X_, Y_ = pickle.load(open(path, 'rb'))
                X = X + [i[:self.input_size] for i in X_]
                Y = Y + Y_
        print(len(X))

        self.dataset = data_utils.TensorDataset(torch.from_numpy(np.asarray(X)), torch.from_numpy(np.asarray(Y)))

    def train(self, checkpoint = None, num_epoch = 100):
        epoch_init = 0
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_init = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            self.model.eval()
            loss.backward()
            self.optimizer.step()
    
    
        for epoch in range(epoch_init, num_epoch + epoch_init):  # loop over the dataset multiple times
            dataloader = data_utils.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)
    
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)
    
                # zero the parameter gradients
                self.optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = self.model(inputs)
                outputs = torch.reshape(outputs, (outputs.size()[0], self.output_size))
    
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
                # print statistics
                running_loss += loss.item()
                if i % 15 == 1:    # print every 5 mini-batches
                    pass
                    #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 15))
                    #running_loss = 0.0

            if epoch % 5 == 0:
                print('[Epoch %d] avg_loss: %.3f' % (epoch + 1, running_loss/len(dataloader)))
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss}, str('checkpoints/' + str(self.model_name) + '_' + str(epoch) + '.pt'))
    
        print('Finished Training')


    def run(self, x):
        y = self.model(torch.tensor(x).float().to(self.device)).detach().cpu().numpy()
        return y


    def num_parameters(self, model):
        parameters = self.model.parameters()
        num_pars = 0
        for par in parameters:
            n_ = 1
            for i in list(par.size()):
                n_ *= i
            num_pars += n_
        return num_pars


            
        


if __name__ == '__main__':
    n = 0
    agent = NeuralNetwork(input_size = (n + 1) * 4, model_name = 'test', batch_size = 1000)#, checkpoint = 'checkpoints/mlp_H10_995.pt')

    agent.data_process(paths = ['expert_traj/expert_pts_10058_H10.p', 'expert_traj/expert_pts_17358_H10.p'])
    agent.train(num_epoch = 1000)
