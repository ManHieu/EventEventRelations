import datetime
import torch
from sklearn.metrics import confusion_matrix

CUDA = torch.cuda.is_available()
# Padding function
def padding(sent, pos = False, max_sent_len = 120):
    if pos == False:
        one_list = [1] * max_sent_len
        one_list[0:len(sent)] = sent
        return torch.tensor(one_list, dtype=torch.long)
    else:
        one_list = ["None"] * max_sent_len
        one_list[0:len(sent)] = sent
        return one_list

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def metric(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    Acc, P, R, F1, _ = CM_metric(CM)
    
    return Acc, P, R, F1, CM

def CM_metric(CM):
    all_ = CM.sum()
    
    Acc = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2] + CM[3][3]) / all_
    P = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0][0:3].sum() + CM[1][0:3].sum() + CM[2][0:3].sum() + CM[3][0:3].sum())
    R = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0].sum() + CM[1].sum() + CM[2].sum())
    F1 = 2 * P * R / (P + R)
    
    return Acc, P, R, F1, CM
