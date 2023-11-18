import torch 
import torch.nn as nn
import torch.nn.functional as F

from settings import settings 

def to_string(y):
    # find EOS index 
    eos_idx = torch.where(y == settings.CHAR2IDX[settings.EOS])[0]
    # remove everything after settings.EOS token
    y = y[:eos_idx[0]] if len(eos_idx) > 0 else y
    # convert to string
    return ''.join([settings.IDX2CHAR[i.item()] for i in y])

def autoregressive_loss(y_pred, y_true):
    loss = 0
    loss_fn = nn.NLLLoss(reduction='sum').to(settings.DEVICE)
    for i in range(y_pred.shape[1]):
        loss += loss_fn(y_pred[:, i, :], y_true[:, i + 1].to(torch.long)) # ignore settings.SOS token
    # normalize loss
    loss /= y_true.shape[0] 
    return loss 

def CTC_loss(y_pred, y_true):
    ctc_loss_fn = nn.CTCLoss(blank=settings.CHAR2IDX[settings.PAD], reduction='sum').to(settings.DEVICE)
    y_pred = F.log_softmax(y_pred, dim=-1)

    input_lengths = torch.IntTensor([y_pred.shape[0]] * y_pred.shape[1]).to(settings.DEVICE)
    target_lengths = torch.IntTensor([torch.sum(y != settings.CHAR2IDX[settings.PAD]) for y in y_true]).to(settings.DEVICE)

    loss = ctc_loss_fn(y_pred, y_true, input_lengths, target_lengths)
    loss /= y_true.shape[0] 

    return loss

def accuracy_fn(y_pred, y_true, decode_fn): 
    y_pred = decode_fn(y_pred)
    tot_ch, ch_acc, sq_acc = 0, 0, 0    
    for i in range(y_pred.shape[0]):
        y_pred_str = to_string(y_pred[i])
        y_true_str = to_string(y_true[i, 1:])
        # total number of characters
        tot_ch += len(y_true_str)
        # character level accuracy
        ch_acc += sum([1 if y_pred_str[j] == y_true_str[j] else 0 for j in range(min(len(y_pred_str), len(y_true_str)))])
        # sequence level accuracy
        sq_acc += 1 if y_pred_str == y_true_str else 0
    # normalize results
    ch_acc /= tot_ch
    sq_acc /= y_pred.shape[0]

    return ch_acc, sq_acc

def autoregressive_decode(y_pred):
    return torch.argmax(y_pred, dim=-1) # (batch, seq_len)

def CTC_decode(y_pred):
    # greedy decoder
    y_pred = torch.argmax(y_pred, dim=-1) # (sequence, batch)
    y_pred = y_pred.permute(1, 0)         # (batch, sequence)

    for i in range(y_pred.shape[0]):
        y_dec = torch.unique_consecutive(y_pred[i])
        y_dec = y_dec[y_dec != settings.CHAR2IDX[settings.PAD]]
        y_pred[i, :len(y_dec)] = y_dec
        y_pred[i, len(y_dec):] = settings.CHAR2IDX[settings.PAD]
    
    # ignore SOS token 
    y_pred = y_pred[:, 1:]

    return y_pred