import torch 
def accuracy(y_pred, y_train):
    y_pred = torch.argmax(y_pred, dim=-1) # (batch, len(CHAR2IDX))
    y_train = y_train[:, 1:] # ignore <SOS> from target
    # compute character accuracy
    chr_acc = torch.sum(y_pred == y_train).item() / torch.numel(y_train)
    # compute sequence accuracy
    seq_acc = torch.sum(torch.all(y_pred == y_train, dim=-1)).item() / y_pred.shape[0]
    # result 
    return chr_acc, seq_acc