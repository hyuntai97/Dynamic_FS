'''
"Dynamic Feature-Selection" Code base 

Code author: Hyuntae Kim (soodaman97@cau.ac.kr), Hyeryn Park (qkrgpfls1201@gmail.com)
----------------------------------------------------------

run.py

(1) Set arguments 

(2) Execute EXP class 
'''

import argparse
import os
import torch
from exp.exp_main_fs2 import Exp_Main_Fs2

import random 
import numpy as np

def main():
    fix_seed = 0
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Dynamic Feature Selection for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='LSTM_FS_final',
                        help='model name, options: [LSTM_FS_final, LSTM]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETRI', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/mnt/storage/projects/etri-beam-management/BM-0006/', help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=7, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=3, help='start token length')
    parser.add_argument('--pred_len', type=int, default=3, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=256, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=256, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=256, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # LSTM & GRU
    parser.add_argument('--num_layers', type=int, default=3, help='num of layers')
    parser.add_argument('--hid_dim', type=int, default=256, help='hidden dimension')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='baseline learning rate')
    parser.add_argument('--learning_rate2', type=float, default=0.00001, help='selector learning rate')
    parser.add_argument('--reg', action='store_true', default=False, help='l1-regularization')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type0', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=6, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # data preprocessing  
    parser.add_argument('--inverse', type=int, default=1, help='1: inverse / 0: non inverse')
    parser.add_argument('--scale_type', type=str, default='standard', help='[minmax standard] scaler type')
    parser.add_argument('--tanh', type=int, default=0, help='1: minmax scaling -1~1 | 0: 0~1')
    parser.add_argument('--mv_avg', action='store_true', default=False, help='moving average preprocessing')

    # fs
    parser.add_argument('--lamb', type=float, default=1, help='feature selection lambda (regularizer)')
    parser.add_argument('--actor_h_dim', type=int, default=10, help='selector hidden dimension')
    parser.add_argument('--loss_op', type=str, default='total', help='forecasting | total')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main_Fs2

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_fc{}_dt{}_ep{}_Loss{}_reg{}_scaler{}_{}_mv_avg{}_lamb{}_hdim{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.factor,
                args.distil,
                args.train_epochs,
                args.loss, 
                args.reg,
                args.scale_type,
                args.tanh,
                args.mv_avg,
                args.lamb,
                args.actor_h_dim)
            

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_ep{}_CNN{}_fil{}_Loss{}_reg{}_{}_scaler{}_{}_mv_avg{}_mask{}_{}_lamb{}_hdim{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.train_epochs,
                args.use_CNN,
                args.filter_size,
                args.loss, 
                args.reg,
                args.txrx,
                args.scale_type,
                args.tanh,
                args.mv_avg,
                args.mask, 
                args.mask_size,
                args.lamb,
                args.actor_h_dim)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
