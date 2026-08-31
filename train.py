import torch
from torch import nn
from pathlib import Path
import transformer
from tokenizer import WMT14Tokenizer
import loss
import optimize
import data
import checkpoint
from evaluate import calculate_token_accuracy,evaluate_nll
from config import get_model_config,get_training_config

def initialize_parameters(net):
    for parameter in net.parameters():
        if parameter.dim()>1:
            nn.init.xavier_uniform_(parameter)

def build_model(model_config, device):
    net=transformer.Transformer(**model_config)
    return net.to(device)


def train_one_epoch(net,dataset,optimizer,device,pad_id,epoch_idx,global_step,
                    max_steps,num_hiddens,warmup_steps,max_tokens,pool_size,seed,
                    log_every_steps,save_every_steps,path,best_nll,model_config,
                    training_history,num_batches,batch_offset=0):
    net.train()
    use_bf16= device.type=="cuda" and torch.cuda.is_bf16_supported()
    dataset.set_epoch(epoch_idx)

    example_batches=data.batch_by_lens(iter(dataset),max_tokens,pool_size,
                                         seed+epoch_idx)
    token_loss=0.0
    total_correct=0
    total_tokens=0
    total_step_batches=0

    for idx,examples in enumerate(example_batches):
        if idx<batch_offset:
            continue

        (src_batch,src_valid_lens,decoder_batch,_,
         label_batch)=data.collate_batch(examples,pad_id)

        src_batch=src_batch.to(device)
        src_valid_lens=src_valid_lens.to(device)
        decoder_batch=decoder_batch.to(device)
        label_batch=label_batch.to(device)

        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_bf16):
            logits=net(src_batch,decoder_batch,src_valid_lens)
            l=loss.label_smooth_cross_entropy(logits,label_batch,pad_id,0.1)
        (l/num_batches).backward()
        total_step_batches+=1

        correct,token_cnt=calculate_token_accuracy(logits,label_batch,pad_id)
        token_loss+=l.item()*token_cnt
        total_correct+=correct
        total_tokens+=token_cnt

        if total_step_batches<num_batches:
            continue

        global_step+=1
        lr=optimize.update_learning_rate(optimizer,global_step,num_hiddens,
                                         warmup_steps)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_step_batches=0

        if global_step==1 or global_step%log_every_steps==0:
            log_loss=token_loss/total_tokens
            log_acc=total_correct/total_tokens
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"step={global_step},loss={log_loss:.4f},accuracy={log_acc:.2%},"
                      f"lr={lr:.8f},tokens={total_tokens}"f",peak_memory={peak_memory:.2f}GiB")

        if global_step%save_every_steps==0:
            checkpoint.save_checkpoint(path,net,optimizer,epoch_idx,global_step,
                                       best_nll,model_config,training_history,False,idx+1)
            print(f"save at global step {global_step}")

        if global_step>=max_steps:
            break

    if total_step_batches>0 and global_step<max_steps:
        w=num_batches/total_step_batches
        for param in net.parameters():
            if param.grad is not None:
                param.grad.mul_(w)
        global_step+=1
        optimize.update_learning_rate(optimizer,global_step,num_hiddens,warmup_steps)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    average_loss=token_loss/total_tokens

    return (global_step,average_loss)


def main():
    seed=42
    model_size="base"
    training_config=get_training_config(model_size)


    max_steps=training_config["max_steps"]
    warmup_steps=training_config["warmup_steps"]
    max_examples=training_config["max_examples"]
    valid_max_examples=training_config["valid_max_examples"]
    max_tokens=training_config["max_tokens"]
    buffer_size=training_config["buffer_size"]
    pool_size=training_config["pool_size"]
    token_budget=training_config["token_budget"]
    log_every_steps=training_config["log_every_steps"]
    save_every_steps=training_config["save_every_steps"]
    num_batches=training_config.get("gradient_accumulation_steps",1)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root=Path(__file__).resolve().parent
    tokenizer_path=(project_root/"vocab"/"wmt14_en_de_bpe_37k.model")
    train_src_path=Path.home()/"datasets"/"wmt14"/"processed"/"plain"/"train.shuffled.en"
    train_tgt_path=Path.home()/"datasets"/"wmt14"/"processed"/"plain"/"train.shuffled.de"
    valid_src_path=Path.home()/"datasets"/"wmt14"/"processed"/"plain"/"valid.en"
    valid_tgt_path=Path.home()/"datasets"/"wmt14"/"processed"/"plain"/"valid.de"
    checkpoint_directory=(project_root/"checkpoints")
    history_checkpoint_directory=(checkpoint_directory/"history")
    last_checkpoint_path=(checkpoint_directory/f"{model_size}_last.pt")
    best_checkpoint_path=(checkpoint_directory/f"{model_size}_best.pt")

    resume_path=(last_checkpoint_path if last_checkpoint_path.exists() else None)
    tokenizer=WMT14Tokenizer(tokenizer_path)
    model_config=get_model_config(model_size,tokenizer.vocab_size)
    train_dataset=data.WMT14Dataset(train_src_path,train_tgt_path,tokenizer,
                                    max_examples,max_tokens,buffer_size,seed)
    valid_dataset=data.WMT14Dataset(valid_src_path,valid_tgt_path,tokenizer,
                                    valid_max_examples,max_tokens,0,seed)
    
    net=build_model(model_config,device)
    initialize_parameters(net)
    optimizer=optimize.optimizer(net)

    batch_offset=0
    start_epoch=0
    global_step=0
    best_validation_nll=float("inf")

    epoch_losses=[]
    validation_nlls=[]
    training_history={"epoch_losses":epoch_losses,
                      "validation_nlls": validation_nlls}
    
    if resume_path is not None:
        checkpoint_data=checkpoint.load_checkpoint(resume_path,net,optimizer,device)

        saved_epoch=checkpoint_data["epoch"]

        if checkpoint_data["epoch_complete"]:
            start_epoch=saved_epoch+1
            batch_offset=0
        else:
            start_epoch=saved_epoch
            batch_offset=checkpoint["batch_offset"]

        global_step=checkpoint_data["global_step"]
        best_validation_nll=checkpoint_data["best_validation_nll"]
        training_history=checkpoint_data["training_history"]

        epoch_losses=list(training_history["epoch_losses"])
        validation_nlls=list(training_history["validation_nlls"])
    
    epoch_idx=start_epoch
    while global_step<max_steps:
        global_step,epoch_loss=train_one_epoch(net,train_dataset,optimizer,device,
                                               tokenizer.pad_id,epoch_idx,
                                               global_step,max_steps,
                                               model_config["num_hiddens"],
                                               warmup_steps,token_budget,pool_size,
                                               seed,log_every_steps,save_every_steps,
                                               last_checkpoint_path,
                                               best_validation_nll,model_config,
                                               training_history,num_batches,batch_offset)
        epoch_losses.append(epoch_loss)
        print(
                f"epoch={epoch_idx+1} completed, "
                f"global_step={global_step}, "
                f"training_loss={epoch_loss:.4f}")
        validation_nll,validation_ppl,validation_acc=evaluate_nll(net,valid_dataset,
                                                                  device,tokenizer.pad_id,
                                                                  token_budget,pool_size,seed)
        validation_nlls.append(validation_nll)
        print(
                f"validation_nll={validation_nll:.4f}, "
                f"perplexity={validation_ppl:.2f}, "
                f"accuracy={validation_acc:.2%}")
        training_history={"epoch_losses":epoch_losses,
                          "validation_nlls":validation_nlls}
        
        is_best=validation_nll<best_validation_nll
        if is_best:
            best_validation_nll=validation_nll
            checkpoint.save_checkpoint(best_checkpoint_path,net,optimizer,epoch_idx,
                                       global_step,best_validation_nll,model_config,
                                       training_history,True)

        checkpoint.save_checkpoint(last_checkpoint_path,net,optimizer,epoch_idx,
                                   global_step,best_validation_nll,model_config,
                                   training_history,True)

        history_checkpoint_path=(history_checkpoint_directory/(
            f"{model_size}_checkpoint_epoch_{epoch_idx + 1:04d}"f"_step_{global_step:08d}.pt"))
        checkpoint.save_model_checkpoint(history_checkpoint_path,net,epoch_idx,
                                         global_step,validation_nll,model_config)
        epoch_idx+=1
        batch_offset=0

if __name__=="__main__":
    main()