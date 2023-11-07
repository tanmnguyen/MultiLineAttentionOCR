import torch 
import torch.nn as nn
import torch.nn.functional as F

from constants import DEVICE, PAD, CHAR2IDX
def CTC_loss(y_pred, y_true):
    ctc_loss_fn = nn.CTCLoss(blank=CHAR2IDX[PAD]).to(DEVICE)

    y_pred = F.log_softmax(y_pred, dim=-1)

    input_lengths = torch.IntTensor(
        [y_pred.shape[0]] * y_pred.shape[1]
    ).to(DEVICE)

    target_lengths = torch.IntTensor(
        [torch.sum(y != CHAR2IDX[PAD]) for y in y_true]
    ).to(DEVICE)
        
    target = y_true
    loss = ctc_loss_fn(
      y_pred,
      target,
      input_lengths,
      target_lengths
    )

    assert loss.item() >= 0.0

    return loss

def CTC_decode(y_pred):
  # greedy decoder
    y_pred = torch.argmax(y_pred, dim=-1) # (sequence, batch)
    y_pred = y_pred.permute(1, 0)         # (batch, sequence)

    decoded_seqs = []
    for pred in y_pred:
        seq = []
        for i in range(len(pred)):
            if pred[i] != CHAR2IDX[PAD] and (i == 0 or pred[i] != pred[i-1]):
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