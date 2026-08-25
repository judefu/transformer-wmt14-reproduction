import torch
from pathlib import Path

def save_checkpoint(path,net,optimizer,epoch,global_step,best_validation_nll,
                    model_config,training_history):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_data = {
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_nll": best_validation_nll,
        "model_config": model_config,
        "training_history": training_history,
    }
    torch.save(checkpoint_data,path)

    return path


def load_checkpoint(path,net,optimizer,device):
    path = Path(path)
    checkpoint_data = torch.load(path,map_location=device,weights_only=False)

    net.load_state_dict(checkpoint_data["net_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    return checkpoint_data