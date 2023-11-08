import torch 
import torch.nn as nn
import torch.nn.functional as F

from constants import DEVICE, PAD, CHAR2IDX, SOS, EOS

def l2_regularization(model, l2_weight=1e-6):
    l2_reg = torch.tensor(0.).to(DEVICE)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    return l2_weight * l2_reg

def autoregressive_loss(y_pred, y_true):
    loss = 0
    loss_fn = nn.NLLLoss(reduction='sum').to(DEVICE)
    for i in range(y_pred.shape[1]):
        loss += loss_fn(y_pred[:, i, :], y_true[:, i + 1].to(torch.long)) # ignore SOS token
    # normalize loss
    loss /= y_true.shape[0] 
    return loss 

def decoder_accuracy_fn(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=-1) # (batch, seq_len)
    tot_ch, ch_acc, sq_acc = 0, 0, 0
    for i in range(y_pred.shape[0]):
        eq = y_pred[i, :] == y_true[i, 1:]  # ignore SOS token
        tot_ch += y_true.shape[1] - 1       # ignore SOS token
        ch_acc += torch.sum(eq).item()
        sq_acc += 1 if torch.sum(eq) == y_pred.shape[1] else 0
    # normalize results
    return ch_acc / tot_ch, sq_acc / y_pred.shape[0]

def CTC_loss(y_pred, y_true):
    ctc_loss_fn = nn.CTCLoss(blank=CHAR2IDX[PAD], reduction='sum').to(DEVICE)

    y_pred = F.log_softmax(y_pred, dim=-1)
    target = y_true

    input_lengths = torch.IntTensor([y_pred.shape[0]] * y_pred.shape[1]).to(DEVICE)
    target_lengths = torch.IntTensor([torch.sum(y != CHAR2IDX[PAD]) for y in y_true]).to(DEVICE)
        
    loss = ctc_loss_fn(
      y_pred,
      target,
      input_lengths,
      target_lengths
    )

    loss /= y_true.shape[0] # normalize loss

    return loss

def CTC_decode(y_pred):
    # greedy decoder
    y_pred = torch.argmax(y_pred, dim=-1) # (sequence, batch)
    y_pred = y_pred.permute(1, 0)         # (batch, sequence)

    decoded_seqs = []
    for pred in y_pred:
        seq = []
        for i in range(len(pred)):
            if  pred[i] != CHAR2IDX[PAD] and (i == 0 or pred[i] != pred[i-1]):
                seq.append(pred[i])
        decoded_seqs.append(seq)

    return decoded_seqs

def CTC_accuracy_fn(y_pred, y_true):
    y_pred = CTC_decode(y_pred)
    tot_ch, ch_acc, sq_acc = 0, 0, 0
    for i in range(y_true.shape[0]):
        pred, true = y_pred[i], y_true[i]
        true = true[true != CHAR2IDX[PAD]] # remove padding
        tot_ch += len(true)
        if len(pred) == len(true):
            cnt = 0
            # update character level accuracy
            for j in range(len(true)):
                cnt += 1 if pred[j] == true[j] else 0
            ch_acc += cnt
            # update sequence level accuracy
            sq_acc += 1 if cnt == len(true) else 0
    # normalize results
    return ch_acc / tot_ch, sq_acc / y_true.shape[0]