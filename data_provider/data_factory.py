'''
"Dynamic Feature-Selection" Code base 

Code author: Hyeryn Park (qkrgpfls1201@gmail.com), Hyuntae Kim (soodaman97@cau.ac.kr)
----------------------------------------------------------

data_factory.py

(1) Data provider function 
'''


from data_provider.data_loader import Dataset_ETRI
from torch.utils.data import DataLoader

data_dict = {
    'ETRI': Dataset_ETRI, 
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size = args.label_len, 
        in_w = args.seq_len, 
        out_w = args.pred_len,
        scale_type=args.scale_type,
        tanh=args.tanh,
    )

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = len(data_set)
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
