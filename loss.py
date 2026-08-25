from torch.nn import functional as F

def label_smooth_cross_entropy(logits,labels,pad_id,smoothing=0.1):
    real_tokens=(labels.reshape(-1)!=pad_id).sum()+1e-6
    loss_sum=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                             labels.reshape(-1),ignore_index=pad_id,
                             label_smoothing=smoothing,reduction="sum")
    return loss_sum/real_tokens