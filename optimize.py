import torch

def learning_rate(step,num_hiddens,warmup_steps=4000):
    return (num_hiddens** -0.5) * min(step** -0.5,step*(warmup_steps** -1.5))

def optimizer(net):
    return torch.optim.Adam(net.parameters(),lr=0,betas=(0.9,0.98),eps=1e-9)

def update_learning_rate(optimizer,step,num_hiddens,warmup_steps=4000):
    lr=learning_rate(step,num_hiddens,warmup_steps)
    for parameter in optimizer.param_groups:
        parameter["lr"]=lr
    return lr