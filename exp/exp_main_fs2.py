'''
"Dynamic Feature-Selection" Code base 

Code author: Hyuntae Kim (soodaman97@cau.ac.kr), Hyeryn Park (qkrgpfls1201@gmail.com)
----------------------------------------------------------

exp_main_fs2.py

(1) EXP class 
    - train 
    - valid 
    - test 

(2) Data preprocessing 
    - make_mask 
    - make_sequence
    - make_sequence_2
'''


from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import LSTM_FS_final
from utils.tools import EarlyStopping, adjust_learning_rate, adjust_learning_rate2, visual
from utils.metrics import metric
from data_provider.data_loader import moving_avg

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

def make_mask(data, in_w, tx, rx):
    mask_list = []
    for i in range(in_w):

        mask_one = np.ones(i+1+(10-in_w))
        if (in_w-i-1) < 0:
            mask_zero = np.zeros(0)
        else:
            mask_zero = np.zeros(in_w-i-1)
        mask = np.hstack([mask_one, mask_zero])
        mask_list.append(mask)

    mask = np.array(mask_list)
    mask = np.tile(mask, (data.shape[0]//7, 1))

    masks = np.repeat(np.expand_dims(np.repeat(np.expand_dims(mask, axis=2), tx, axis=2), axis=3), rx, axis=3)
    masks_ = torch.FloatTensor(masks)
    arr_mask = data * masks_    

    return arr_mask

def make_sequence(seq, in_w, out_w):

    seq_x = seq[:, :in_w, :, :]
    seq_y = seq[:, in_w:, : :]

    return seq_x, seq_y

def make_sequence_2(seq, in_w, out_w):

    for sample in range(seq.shape[0]):
        for ti in range(seq.shape[1]-out_w):
            if ti == 0:
                seq_x = seq[sample, ti:ti+in_w, :, :]
                seq_y = seq[sample, ti+in_w:ti+in_w+out_w, : :]
            
            else:
                seq_x = torch.cat((seq_x, seq[sample, ti:ti+in_w, :, :]),0)
                seq_y = torch.cat((seq_y, seq[sample, ti+in_w:ti+in_w+out_w, : :]),0)
            
        if sample == 0:
            seq_x_ = seq_x
            seq_y_ = seq_y
        
        else:
            seq_x_ = torch.cat((seq_x_, seq_x),0)
            seq_y_ = torch.cat((seq_y_, seq_y),0)

    return seq_x_, seq_y_


class Exp_Main_Fs2(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_Fs2, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'LSTM_FS_final':LSTM_FS_final,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        m_params = list(self.model.p_lstm.parameters()) + list(self.model.p_fc.parameters()) + list(self.model.s_lstm.parameters()) + list(self.model.s_fc.parameters())
        s_params = list(self.model.fs.parameters())
        model_optim = optim.Adam(m_params, lr=self.args.learning_rate)
        select_optim = optim.Adam(s_params, lr=self.args.learning_rate2)
        return model_optim, select_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterion = nn.MSELoss(reduction='none', reduce=False)
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss()
            
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch) in enumerate(vali_loader):

                batch_ = batch.float()
                batch = batch_.reshape(batch_.shape[0], batch_.shape[1], -1)

                batch_x = batch[:,:9,:].to(self.device)
                batch_y = batch[:,2:,:].to(self.device)

                (mask, base, outputs, reg, mus, probs, z) = self.model(batch_x)

                if self.args.scale_type == 'minmax':
                    if self.args.tanh == 0:
                        outputs = torch.sigmoid(outputs)
                    else:
                        outputs = torch.tanh(outputs)

                base = criterion(base, batch[:,1:9,:].to(self.device))
                loss_ = criterion(outputs, batch[:,2:,:].to(self.device)) # b,8,256
                if self.args.loss_op == 'forecasting':
                    model_loss = loss_ + self.args.lamb * reg.to(self.device)
                elif self.args.loss_op == 'total':
                    model_loss = (1-mask) * base + loss_ + self.args.lamb * reg.to(self.device)
                loss = model_loss.mean(axis=-1).sum(axis=-1).mean(axis=-1)  

                loss = loss.detach().cpu()
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, select_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                select_optim.zero_grad()
                batch_ = batch_.float()
                batch = batch_.reshape(batch_.shape[0], batch_.shape[1], -1)
                
                (mask, base, predict, reg, mus, probs, z) = self.model(batch[:,:9,].to(self.device))

                base = criterion(base, batch[:,1:9,:].to(self.device))
                loss_ = criterion(predict, batch[:,2:,:].to(self.device)) # b,8,256
                if self.args.loss_op == 'forecasting':
                    model_loss = loss_ + self.args.lamb * reg.to(self.device)
                elif self.args.loss_op == 'total':
                    model_loss = (1-mask) * base + loss_ + self.args.lamb * reg.to(self.device)
                loss = model_loss.mean(axis=-1).sum(axis=-1).mean(axis=-1)  # pair -> time -> sample 순서로

                train_loss.append(loss.item())

                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.step(select_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    select_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            adjust_learning_rate2(select_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        preds = []
        trues = []
        masks = []
        probs = []
        inputs = []

        #folder_path = '/mnt/storage/personal/hrpark/ETRI_BEAM/result/' + setting + '/'
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_) in enumerate(test_loader):
                batch_ = batch_.float()
                batch = batch_.reshape(batch_.shape[0], batch_.shape[1], -1)

                batch_x = batch[:,:9,:].to(self.device)
                batch_y = batch[:,2:,:].to(self.device)

                (mask, _, outputs, reg, mus, prob, z) = self.model(batch_x)

                outputs = outputs.detach().cpu()
                mask = mask.detach().cpu()
                reg = reg.detach().cpu()
                batch_x = batch_x.detach().cpu()

                if self.args.scale_type == 'minmax':
                    if self.args.tanh == 0:
                        outputs = torch.sigmoid(outputs)
                    else:
                        outputs = torch.tanh(outputs)

                # Inverse Transform
                if self.args.inverse == 1:
                    if self.args.tanh == 0:
                        outputs = test_data.inverse_transform(outputs)
                        batch_y = test_data.inverse_transform(batch_y)
                        batch_x = test_data.inverse_transform(batch_x)
                    else:
                        outputs = (outputs + 1) / 2
                        batch_y = (batch_y + 1) / 2
                        batch_x = (batch_x + 1) / 2
                        outputs = test_data.inverse_transform(outputs)
                        batch_y = test_data.inverse_transform(batch_y)
                        batch_x = test_data.inverse_transform(batch_x)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()
                prob = prob.detach().cpu().numpy()

                pred = outputs.reshape(outputs.shape[0], 8, -1)
                true = batch_y.reshape(outputs.shape[0], 8, -1)
                input = batch_x.reshape(batch_x.shape[0], 8, -1)

                preds.append(pred)
                trues.append(true)
                masks.append(mask)
                probs.append(prob)
                inputs.append(input)

                if i % 20 == 0:
                    visual(true[0, :, -1], pred[0, :, -1], os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = np.concatenate(inputs, axis=0)
        probs = np.concatenate(probs, axis=0)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        masks = np.concatenate(masks, axis=0)
        print('mask shape:', masks.shape)

        # result save
        folder_path = './result/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        metric(setting, preds, trues, masks, folder_path)

        for k in range(50):
            plt.figure(figsize=(20,5))
            plt.title('mask{} prob'.format(k))
            plt.imshow(probs[k])
            plt.show()
            plt.savefig(folder_path+'prob_{}_{}'.format(k, setting)+'.png')
            plt.close()
            plt.figure(figsize=(20,5))
            plt.title('sample{} prob'.format(k))
            plt.imshow(inputs[k])
            plt.show()
            plt.savefig(folder_path+'sample_{}_{}'.format(k, setting)+'.png')
            plt.close()

        return