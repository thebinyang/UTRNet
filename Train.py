"""
UTRnet-Train
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
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import torch.nn.functional as F
import crnn
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

def FWCLoss( pred,target, W, count_pos, count_neg):
    m = nn.Sigmoid()
    lossinput = m(pred)
    ratio= count_neg/(count_pos + count_neg)
    W = W.unsqueeze(1).expand_as(target)
    L = - (ratio * target * torch.log(lossinput + 1e-10) * torch.pow(input=(1-lossinput),exponent=2) +
           (1 - ratio) * (1 - target) * torch.log(1 - lossinput + 1e-10) * torch.pow(input=lossinput,exponent=2))
    output = torch.mean(L * W)
    return output

def seed_everything(seed, cuda=True):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)



class ConvDisRNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, time_len):
        super(ConvDisRNN, self).__init__()
        f = open('Parameter.yaml', 'r').read()
        cfig = yaml.load(f)
        self.dislstmcell = crnn.LSTMdistCell(cfig['mode'], 64, 64, kernel_size, convndim=2)
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
                hx, cx = self.dislstmcell(t, [time_dis[:, i-1], time_dis[:, i]], (hx, cx))

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
        if self.cfig['mode'] in ['LSTM','TimeLSTM','Distanced_LSTM','UTRNet']:
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

    def train(self):
        print('Training..................')
        path_train = self.cfig['data_path']
        with open(os.path.join(path_train, 'all_sample.pickle'), 'rb') as file:
            all_sample = pickle.load(file)
        with open(os.path.join(path_train, 'all_label.pickle'), 'rb') as file:
            all_label = pickle.load(file) # 0 is unchanged, 1 is changed
        with open(os.path.join(path_train, 'all_W.pickle'), 'rb') as file:
            all_W = pickle.load(file)

        all_sample = torch.from_numpy(all_sample)
        all_sample = all_sample.permute([0, 1, 4, 2, 3])
        all_sz = all_label.shape[0]
        all_dist = np.zeros((all_sz, 10))
        all_dist[:, :] = self.cfig['image_time']
        all_dist = torch.from_numpy(all_dist)
        model_root = osp.join(self.cfig['save_path'], 'models')


        all_idx = np.arange(0, all_sz)
        np.random.seed(1)
        np.random.shuffle(all_idx)
        train_idx = all_idx[0:int(0.8 * all_sz)]
        val_idx = all_idx[int(0.8 * all_sz):all_sz]
        train_sample = all_sample[:, train_idx, :, :, :]
        val_sample = all_sample[:, val_idx, :, :, :]
        train_W = all_W[train_idx]
        val_W = all_W[val_idx]
        train_label = all_label[train_idx]
        val_label = all_label[val_idx]

        best_loss = 2000
        BATCH_SZ = self.cfig['batch_size']
        iter_train_epoch = train_label.size // BATCH_SZ
        iter_val_epoch = val_label.size // BATCH_SZ

        train_dist = all_dist[train_idx, :]
        val_dist = all_dist[val_idx, :]

        val_Loss_list = []
        val_Accuracy_list = []

        for epoch in tqdm(range(self.cfig['start_epoch'], self.cfig['max_epoch'])):
            train_ave_loss = 0
            train_ave_accuracy = 0
            train_ave_f1 = 0
            val_ave_loss = 0
            val_ave_accuracy = 0
            val_ave_f1 = 0
            if self.cfig['adjust_lr']:
                self.adjust_learning_rate(self.optim, epoch, self.cfig['steps'], self.cfig['lr_gamma'])
                self.optim = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999))

            train_idx = np.arange(0, train_idx.size)
            np.random.seed(1)
            np.random.shuffle(train_idx)
            val_idx = np.arange(0, val_idx.size)
            np.random.seed(1)
            np.random.shuffle(val_idx)

            for _iter in range(iter_train_epoch):
                start_idx = _iter * BATCH_SZ
                end_idx = (_iter + 1) * BATCH_SZ
                batch_train = train_sample[:, train_idx[start_idx:end_idx], :, :, :]
                batch_label = train_label[train_idx[start_idx:end_idx]]
                batch_W = train_W[train_idx[start_idx:end_idx]]
                batch_dist = train_dist[train_idx[start_idx:end_idx]]
                batch_train = torch.as_tensor(batch_train, dtype=torch.float32)
                batch_label = torch.as_tensor(batch_label, dtype=torch.long)
                batch_dist = torch.as_tensor(batch_dist, dtype=torch.float32)

                batch_W = torch.as_tensor(batch_W, dtype=torch.float32)
                batch_train, batch_label, batch_dist, batch_W = batch_train.to(
                    self.device), batch_label.to(
                    self.device), batch_dist.to(self.device), batch_W.to(self.device)
                train_epoch_loss, train_accuracy, train_f1 = self.train_epoch(batch_train, batch_label,batch_dist, batch_W)

                train_ave_loss += train_epoch_loss
                train_ave_accuracy += train_accuracy
                train_ave_f1 += train_f1

            for _iter in range(iter_val_epoch):
                start_idx = _iter * BATCH_SZ
                end_idx = (_iter + 1) * BATCH_SZ
                batch_val = val_sample[:, val_idx[start_idx:end_idx], :, :, :]
                batch_label = val_label[val_idx[start_idx:end_idx]]
                batch_W = val_W[val_idx[start_idx:end_idx]]
                batch_dist = val_dist[val_idx[start_idx:end_idx]]
                batch_val = torch.as_tensor(batch_val, dtype=torch.float32)
                batch_label = torch.as_tensor(batch_label, dtype=torch.long)
                batch_dist = torch.as_tensor(batch_dist, dtype=torch.float32)
                batch_W = torch.as_tensor(batch_W, dtype=torch.float32)
                batch_val, batch_label, batch_dist, batch_W = batch_val.to(self.device), batch_label.to(
                    self.device), batch_dist.to(self.device), batch_W.to(self.device)
                val_epoch_loss, val_accuracy, val_f1 = self.val_epoch(batch_val, batch_label, batch_dist, batch_W)
                val_ave_loss += val_epoch_loss
                val_ave_accuracy += val_accuracy
                val_ave_f1 += val_f1

            best_model_last = '%s/best_model_last.pth' % (model_root)

            train_ave_loss /= iter_train_epoch
            train_ave_accuracy /= iter_train_epoch
            train_ave_f1 /= iter_train_epoch

            val_ave_loss /= iter_val_epoch
            val_ave_accuracy /= iter_val_epoch
            val_ave_f1 /= iter_val_epoch

            val_Loss_list.append(val_ave_loss)
            val_Accuracy_list.append(val_ave_accuracy)
            print("train_ave_loss：%.4f    train_ave_accuracy：%.4f  train_ave_f1：%.4f  " % (
                train_ave_loss, train_ave_accuracy, train_ave_f1))
            print("val_ave_loss：%.4f    val_ave_accuracy：%.4f   val_ave_f1：%.4f" % (
                val_ave_loss, val_ave_accuracy, val_ave_f1))

            if val_ave_loss < best_loss:
                best_loss = val_ave_f1
                torch.save(self.model.state_dict(), best_model_last)
                print("best model is saved, index")


    def train_epoch(self, batch_train,batch_label,batch_dist, batch_W):
        self.model.train()

        pred_list, target_list, loss_list, pos_list = [], [], [], []

        self.optim.zero_grad()

        pred = self.model(batch_train, batch_dist)
        # pred_prob = F.softmax(pred, dim=1)
        pred_prob = torch.sigmoid(pred)
        count_pos = torch.sum(batch_label)
        count_neg = torch.sum(1 - batch_label)
        target = torch.eye(2)[batch_label, :].to(self.device) # one-hot
        loss = FWCLoss(pred, target, batch_W, count_pos, count_neg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)
        self.optim.step()

        pred_cls = pred_prob.data.max(1)[1]
        pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()

        pred_list += pred_cls.data.cpu().numpy().tolist()
        target_list += batch_label.data.cpu().numpy().tolist()
        loss_list.append(loss.data.cpu().numpy().tolist())
        train_epoch_loss = loss.item()
        accuracy = accuracy_score(target_list, pred_list)
        f1 = f1_score(target_list, pred_list, pos_label=1)
        return train_epoch_loss, accuracy, f1


    def val_epoch(self, batch_train, batch_label, batch_dist, batch_W):
        self.model.eval()

        pred_list, target_list, loss_list, pos_list = [], [], [], []

        self.optim.zero_grad()

        pred = self.model(batch_train, batch_dist)
        # pred_prob = F.softmax(pred, dim=1)
        pred_prob = torch.sigmoid(pred)
        count_pos = torch.sum(batch_label)
        count_neg = torch.sum(1 - batch_label)
        target = torch.eye(2)[batch_label, :].to(self.device) # one-hot
        loss = FWCLoss(pred, target, batch_W, count_pos, count_neg)

        pred_cls = pred.data.max(1)[1]
        pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
        pred_list += pred_cls.data.cpu().numpy().tolist()
        target_list += batch_label.data.cpu().numpy().tolist()
        loss_list.append(loss.data.cpu().numpy().tolist())
        val_epoch_loss = loss.item()
        OA = accuracy_score(target_list, pred_list)
        f1 = f1_score(target_list, pred_list, pos_label=1)

        return val_epoch_loss, OA, f1






if __name__ == '__main__':
    f = open('Parameter.yaml', 'r').read()
    cfig = yaml.load(f)
    Seed = cfig['Seed']
    seed_everything(seed=Seed, cuda=True)
    trainer = Trainer(cfig)


    trainer.train()
