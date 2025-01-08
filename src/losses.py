import torch as th

def dice_coef(y_true: th.tensor, y_pred: th.tensor):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    
    intersection = (y_true * y_pred).sum()
    
    return (2. * intersection) / (y_true + y_pred).sum()