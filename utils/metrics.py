'''
"Dynamic Feature-Selection" Code base 

Code author: Hyuntae Kim (soodaman97@cau.ac.kr), Hyeryn Park (qkrgpfls1201@gmail.com)
----------------------------------------------------------

metrics.py

(1) metric
    - evaluate using total metrics

(2) top_1_acc 

(3) top_k_acc 
    
(4) avg_rank 
'''

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

# def metric(pred, true):

#     a = mean_absolute_error(pred.reshape(-1), true.reshape(-1))
#     b = mean_squared_error(pred.reshape(-1), true.reshape(-1))
#     c = mean_absolute_percentage_error(pred.reshape(-1), true.reshape(-1))
#     d = r2_score(pred.reshape(-1), true.reshape(-1))

#     return a, b, c, d

def top_1_acc(pred, true):
    top_1 = []
    for i in range(pred.shape[1]):
        pred_ = pred[:, i, :]
        true_ = true[:, i, :]
        top1 = (np.argmax(pred_, axis=1) == np.argmax(true_, axis=1)).sum() / pred.shape[0]
        top_1.append(round(top1,3))
    return top_1

def top_k_acc(k, pred, true):
    top_k_acc = []
    for i in range(pred.shape[1]):
        pred_ = pred[:, i, :]
        true_ = true[:, i, :]
        top1_p = np.argsort(pred_, axis=1)[:, -(1)]

        for k_ in range(k):     # k=1 / 10
            # 각 sample별로 상위 k index
            top10_t = np.argsort(true_, axis=1)[:, -(k_+1)]
            same = np.where(top1_p == top10_t)[0]
            if k_ > 0:
                top = np.concatenate((top, same))
            else:
                top = same

        top = len(list(set(top))) / pred.shape[0]
        top_k_acc.append(round(top,3))
        
    return top_k_acc

def avg_rank(pred, true):
    avg_rank = []
    for i in range(pred.shape[1]):
        pred_ = pred[:, i, :]
        true_ = true[:, i, :]

        best_real = np.argmax(true_, axis=1)    # (각 sample별 best index)
        best_pred = np.argsort(pred_, axis=1) 

        rank = 0
        for i in range(pred.shape[0]):
            rank += 256-np.where(best_pred[i] == best_real[i])[0][0]
        avg_rank.append(round(rank/pred.shape[0],3))
    return avg_rank

def metric(model_name, pred, true, masks, folder_path):

    total_m = []
    a = mean_absolute_error(pred.reshape(-1), true.reshape(-1))
    b = mean_squared_error(pred.reshape(-1), true.reshape(-1))
    c = mean_absolute_percentage_error(pred.reshape(-1), true.reshape(-1))
    d = r2_score(pred.reshape(-1), true.reshape(-1))
    total_m.extend([round(a,3),round(b,3),round(c,3),round(d,3)])

    time_m = []
    for i in range(pred.shape[1]):
        a = mean_absolute_error(pred[:,i,:].reshape(-1), true[:,i,:].reshape(-1))
        b = mean_squared_error(pred[:,i,:].reshape(-1), true[:,i,:].reshape(-1))
        c = mean_absolute_percentage_error(pred[:,i,:].reshape(-1), true[:,i,:].reshape(-1))
        d = r2_score(pred[:,i,:].reshape(-1), true[:,i,:].reshape(-1))
        time_m.extend([round(a,3),round(b,3),round(c,3),round(d,3)])

    top_k_acc_10 = top_k_acc(10, pred, true)
    top_1 = top_1_acc(pred, true)
    avg_rank_ = avg_rank(pred, true)
    m_top_k_acc_10 = round(np.array(top_k_acc_10).mean(),3)
    m_top_1 = round(np.array(top_1).mean(),3)
    m_avg_rank_ = round(np.array(avg_rank_).mean(),3)


    f = open("{}metric.txt".format(folder_path), "a")
    f.write(model_name + "\n")
    f.write('total (mean) metric:{} \n'.format(total_m))
    f.write('time metric:{} \n'.format(time_m))
    f.write('top_10_acc:{} \n'.format(top_k_acc_10))
    f.write('top_1_acc:{} \n'.format(top_1))
    f.write('avg_rank:{} \n'.format(avg_rank_))
    f.write('[mean] top_10_acc:{} '.format(m_top_k_acc_10))
    f.write(' | top_1_acc:{} '.format(m_top_1))
    f.write(' | mean_avg_rank:{} \n'.format(m_avg_rank_))
    f.write('{} '.format(np.round(len(np.where(masks != 1)[0]) / len(masks.reshape(-1)) , 3)))    # prob 1이 아닌것 : 즉 선택되지 않은
    f.write('{} '.format(np.round(len(np.where(masks != 0)[0]) / len(masks.reshape(-1)) , 3)))    # prob 0이 아닌것 : 즉 선택된
    f.write('\n\n')
    f.close()

    return