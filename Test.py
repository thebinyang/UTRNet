"""
UTRnet-Test
@author: LQ
"""
import sys
import time
import matplotlib.pyplot as plt

sys.path.append('../..')
import yaml
import pickle
import os
import os.path as osp
import torch
import torch.nn as nn
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
# from func.tools.logger import Logger
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_auc_score
import torch.nn.functional as F
import crnn
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class ConvDisRNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, time_len):
        super(ConvDisRNN, self).__init__()
        f = open('Parameter.yaml', 'r').read()
        cfig = yaml.load(f)
        self.dislstmcell = crnn.LSTMdistCell(cfig['mode'], 64, 64, kernel_size, convndim=2)
        # self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
        self.time_len = time_len
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=(1, 1))
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.5)  # Dropout layer
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, time_dis):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(self.time_len):

            t = torch.relu(self.dropout(self.conv1(x[i])))

            t = torch.relu(self.dropout(self.conv2(t)))
            t = torch.relu(self.dropout(self.conv3(t)))

            t = torch.relu(self.dropout(self.conv4(t)))
            t = torch.relu(self.dropout(self.conv5(t)))
            t = torch.relu(self.dropout(self.conv5(t)))

            if i == 0:
                hx, cx = self.dislstmcell(t, [time_dis[:, 0], time_dis[:, 0]])
            else:
                hx, cx = self.dislstmcell(t, [time_dis[:, i - 1], time_dis[:, i]], (hx, cx))

        x = hx.contiguous().view(hx.size()[0], -1)
        x = torch.relu(self.fc1(self.dropout(x)))
        x = torch.relu(self.fc2(self.dropout(x)))
        return x


class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig
        self.lr = cfig['learning_rate']
        self.lr2 = cfig['learning_rate_abcd']
        if self.cfig['mode'] in ['LSTM', 'TimeLSTM', 'Distanced_LSTM', 'UTRNet']:
            print('======distance rnn========')
            self.model = ConvDisRNN(in_channels=6, out_channels=64,
                                    kernel_size=3, num_classes=self.cfig['n_classes'],
                                    time_len=self.cfig['time'] - self.cfig['begin_time']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def kappa(confusion_matrix):
        pe_rows = np.sum(confusion_matrix, axis=0)
        pe_cols = np.sum(confusion_matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(confusion_matrix) / float(sum_total)
        return (po - pe) / (1 - pe)

    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch == 0:
            self.lr = self.cfig['learning_rate']
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
                print('After modify, the learning rate is', param_group['lr'])

    def change_map(self):

        print('Testing..................')

        model_root = osp.join(self.cfig['save_path'], 'models')
        best_model_pth = '%s/best_model_last.pth' % (model_root)
        self.model.load_state_dict(torch.load(best_model_pth))  ##Disrnn

        path_test = self.cfig['data_path']
        with open(os.path.join(path_test, 'all_map.pickle'), 'rb') as file:
            all_sample = pickle.load(file)
            all_sample = all_sample[:, :, :, :, 0:6]

        all_sz = all_sample.shape[1]

        allmap = loadmat(path_test + '/' + 'data.mat')['allmap']
        BATCH_SZ = allmap.shape[0]
        iter_in_test = all_sz // BATCH_SZ
        test_idx = np.arange(0, all_sz)

        test_sample = torch.from_numpy(all_sample)
        test_sample = test_sample.permute([0, 1, 4, 2, 3])

        dist = np.zeros((all_sz, 10))
        dist[:, :] = self.cfig['image_time']
        test_dist = torch.from_numpy(dist)
        pred = []

        for _iter in range(iter_in_test):
            start_idx = _iter * BATCH_SZ
            end_idx = (_iter + 1) * BATCH_SZ
            batch_val = test_sample[:, test_idx[start_idx:end_idx], :, :, :]
            batch_dist = test_dist[test_idx[start_idx:end_idx]]
            batch_dist = torch.as_tensor(batch_dist, dtype=torch.float32)
            batch_val = torch.as_tensor(batch_val, dtype=torch.float32)
            batch_val, batch_dist = batch_val.to(self.device), batch_dist.to(self.device)

            pred_list = self.test_all_epoch(batch_val, batch_dist)
            pred.extend(pred_list)

        pred_label = np.array(pred)
        all_map = np.reshape(pred_label, (allmap.shape[0], allmap.shape[1]))

        height, width = all_map.shape
        plt.imshow(all_map, aspect='equal', cmap='gray')
        plt.axis('off')
        plt.gcf().set_size_inches(width / 100, height / 100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('all_map.jpg', transparent=True, dpi=600, pad_inches=0)
        plt.show()

    def change_map_epoch(self, batch_train, batch_dist):
        self.model.eval()

        pred_list, target_list, loss_list, pos_list = [], [], [], []

        self.optim.zero_grad()

        pred = self.model(batch_train, batch_dist)
        pred_prob = torch.sigmoid(pred)
        pred_cls = pred_prob.data.max(1)[1]
        pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
        pred_list += pred_cls.data.cpu().numpy().tolist()

        return pred_list

    def test(self):

        print('Testing..................')

        model_root = osp.join(self.cfig['save_path'], 'models')
        best_model_pth = '%s/best_model_last.pth' % (model_root)
        self.model.load_state_dict(torch.load(best_model_pth))

        path_test = self.cfig['data_path']
        with open(os.path.join(path_test, 'test_sample.pickle'), 'rb') as file:
            test_sample = pickle.load(file)
            test_sample = test_sample[:, :, :, :, 0:6]
        with open(os.path.join(path_test, 'test_label.pickle'), 'rb') as file:
            test_label = pickle.load(file)  # 0 is unchanged, 1 is changed

        test_sz = test_label.shape[0]
        BATCH_SZ = 100
        iter_in_test = test_sz // BATCH_SZ
        test_idx = np.arange(0, test_sz)

        test_sample = torch.from_numpy(test_sample)
        test_sample = test_sample.permute([0, 1, 4, 2, 3])
        test_label = np.reshape(test_label, (-1))
        dist = np.zeros((test_sz, 10))
        dist[:, :] = self.cfig['image_time']
        test_dist = torch.from_numpy(dist)

        pred = []
        real = []

        for _iter in range(iter_in_test):
            start_idx = _iter * BATCH_SZ
            end_idx = (_iter + 1) * BATCH_SZ
            batch_val = test_sample[:, test_idx[start_idx:end_idx], :, :, :]
            batch_label = test_label[test_idx[start_idx:end_idx]]
            batch_dist = test_dist[test_idx[start_idx:end_idx]]
            batch_val = torch.as_tensor(batch_val, dtype=torch.float32)
            batch_label = torch.as_tensor(batch_label, dtype=torch.int64)
            batch_dist = torch.as_tensor(batch_dist, dtype=torch.float32)

            batch_val, batch_dist = batch_val.to(self.device), batch_dist.to(self.device)

            pred_list, target_list = self.test_epoch(batch_val, batch_label, batch_dist)
            pred.extend(pred_list)
            real.extend(target_list)

        OA = accuracy_score(real, pred)
        precision = precision_score(real, pred, pos_label=1)
        recall = recall_score(real, pred, pos_label=1)
        f1 = f1_score(real, pred, pos_label=1)

        Kappa = Trainer.kappa(confusion_matrix(real, pred))
        print(confusion_matrix(real, pred))
        print('precision: %.4f   recall: %.4f  OA: %.4f   F1: %.4f  Kappa: %.4f  ' % (precision, recall, OA, f1, Kappa))

    def test_epoch(self, batch_train, batch_label, batch_dist):
        self.model.eval()

        pred_list, target_list, loss_list, pos_list = [], [], [], []

        self.optim.zero_grad()

        pred = self.model(batch_train, batch_dist)
        pred_prob = torch.sigmoid(pred)
        pred_cls = pred_prob.data.max(1)[1]
        pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
        pred_list += pred_cls.data.cpu().numpy().tolist()
        target_list += batch_label.data.cpu().numpy().tolist()

        return pred_list, target_list





if __name__ == '__main__':
    f = open('Parameter.yaml', 'r').read()
    cfig = yaml.load(f)
    trainer = Trainer(cfig)
    trainer.test()
    # trainer.change_map()  % get the change map
