import torch
from pathlib import Path

def save_checkpoint(path,net,optimizer,epoch,global_step,best_validation_nll,
                    model_config,training_history,epoch_complete=False,batch_offset=0):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)

    checkpoint_data={"net_state_dict":net.state_dict(),
                     "optimizer_state_dict":optimizer.state_dict(),
                     "epoch":epoch,
                     "global_step":global_step,
                     "best_validation_nll":best_validation_nll,
                     "model_config":model_config,
                     "training_history":training_history,
                     "epoch_complete":epoch_complete,
                     "batch_offset":batch_offset}
    torch.save(checkpoint_data,path)
    return path

def save_model_checkpoint(path,net,epoch,global_step,validation_nll,model_config):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)

    checkpoint_data={"checkpoint_type":"model_only",
                     "net_state_dict":net.state_dict(),
                     "epoch":epoch,
                     "global_step":global_step,
                     "validation_nll":validation_nll,
                     "model_config":model_config}
    torch.save(checkpoint_data,path)
    return path

def load_checkpoint(path,net,optimizer,device):
    path=Path(path).expanduser()
    checkpoint_data=torch.load(path,map_location=device,weights_only=False)
    net.load_state_dict(checkpoint_data["net_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    return checkpoint_data

def tensor_storage_signature(tensor):
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )

def average_checkpoints(checkpoint_paths,output_path):
    checkpoint_paths=[Path(path).expanduser().resolve()
                       for path in checkpoint_paths]
    output_path=(Path(output_path).expanduser().resolve())

    added_tensors={}
    shared_params={}
    name_by_storage={}

    real_keys=None
    real_dtypes={}
    is_floating={}
    real_model_config=None

    source_checkpoints=[]

    latest_epoch=None
    latest_global_step=None

    checkpoints_num=len(checkpoint_paths)

    with torch.no_grad():
        for idx, path in enumerate(checkpoint_paths):
            checkpoint_data=torch.load(path,map_location="cpu",weights_only=False)
            state_dict=checkpoint_data["net_state_dict"]

            if idx==0:
                real_keys=list(state_dict.keys())
                real_model_config=checkpoint_data["model_config"]

                for name, tensor in state_dict.items():
                    real_dtypes[name]=tensor.dtype
                    is_floating[name]=torch.is_floating_point(tensor)
                    storage_signature=tensor_storage_signature(tensor)

                    if (storage_signature in name_by_storage):
                        shared_name=name_by_storage[storage_signature]
                        shared_params[name]=shared_name
                        continue

                    name_by_storage[storage_signature]=name

                    if is_floating[name]:
                        if tensor.dtype==torch.float64:
                            add_dtype=torch.float64
                        else:
                            add_dtype=torch.float32

                        added_tensors[name]=tensor.to(dtype=add_dtype).clone()
                    else:
                        added_tensors[name]=tensor.clone()
            else:
                for name, tensor in state_dict.items():
                    if name in shared_params:
                        continue

                    if is_floating[name]:
                        added_tensors[name].add_(tensor.to(
                            dtype=added_tensors[name].dtype))

                    else:
                        added_tensors[name]=tensor.clone()

            latest_epoch=checkpoint_data["epoch"]
            latest_global_step=checkpoint_data["global_step"]

            source_checkpoints.append(
                {"path":str(path),
                 "epoch":latest_epoch,
                 "global_step":latest_global_step,
                "validation_nll":checkpoint_data.get("validation_nll")})
            
            del checkpoint_data
            del state_dict

        for name in added_tensors:
            if is_floating[name]:
                added_tensors[name].div_(checkpoints_num)
                added_tensors[name]=added_tensors[name].to(dtype=real_dtypes[name])

        averaged_state_dict={}

        for name in real_keys:
            if name in shared_params:
                param_name=shared_params[name]
            else:
                param_name=name
            
            averaged_state_dict[name]=added_tensors[param_name]

    averaged_checkpoint={
        "checkpoint_type":"averaged_model",
        "net_state_dict":averaged_state_dict,
        "model_config":real_model_config,
        "epoch":latest_epoch,
        "global_step":latest_global_step,
        "best_validation_nll":None,
        "num_averaged_checkpoints":checkpoints_num,
        "source_checkpoints":source_checkpoints}

    output_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(averaged_checkpoint,output_path)

    return output_path

def main():
    project_directory=Path(__file__).resolve().parent

    model_size="base"
    num_checkpoints=5

    history_directory=project_directory/"checkpoints"/"history"
    output_path=project_directory/"checkpoints"/f"{model_size}_averaged_last_{num_checkpoints}.pt"
    pattern=(f"{model_size}_checkpoint_epoch_"f"*_step_*.pt")
    checkpoint_paths=sorted(history_directory.glob(pattern))

    selected_paths=checkpoint_paths[-num_checkpoints:]
    average_checkpoints(selected_paths,output_path)

if __name__ == "__main__":
    main()