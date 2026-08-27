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
                    training_history):
    net.train()
    dataset.set_epoch(epoch_idx)

    example_batches=data.batch_by_lens(iter(dataset),max_tokens,pool_size,
                                         seed+epoch_idx)
    token_loss=0.0
    total_correct=0
    total_tokens=0

    for examples in example_batches:
        (src_batch,src_valid_lens,decoder_batch,_,
         label_batch)=data.collate_batch(examples,pad_id)

        src_batch=src_batch.to(device)
        src_valid_lens=src_valid_lens.to(device)
        decoder_batch=decoder_batch.to(device)
        label_batch=label_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits=net(src_batch,decoder_batch,src_valid_lens)
        l=loss.label_smooth_cross_entropy(logits,label_batch,pad_id,0.1)
        l.backward()
        global_step+=1
        lr=optimize.update_learning_rate(optimizer,global_step,num_hiddens,
                                         warmup_steps)
        optimizer.step()

        correct,token_cnt=calculate_token_accuracy(logits,label_batch,pad_id)
        token_loss+=l.item()*token_cnt
        total_correct+=correct
        total_tokens+=token_cnt

        if global_step==1 or global_step%log_every_steps==0:
            log_loss=token_loss/total_tokens
            log_acc=total_correct/total_tokens            
            print(f"step={global_step},loss={log_loss:.4f},accuracy={log_acc:.2%},"
                      f"lr={lr:.8f},tokens={total_tokens}")

        if global_step%save_every_steps==0:
            checkpoint.save_checkpoint(path,net,optimizer,epoch_idx,global_step,
                                       best_nll,model_config,training_history)
            
        if global_step>=max_steps:
            break

    average_loss=token_loss/total_tokens
        
    return (global_step,average_loss)


def main():
    seed=42
    model_size="tiny"
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

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root=Path(__file__).resolve().parent
    tokenizer_path=(project_root/"vocab"/"wmt14_en_de_bpe_37k.model")
    train_src_path=Path("~/datasets/wmt14/processed/plain/train.en").expanduser()
    train_tgt_path=Path("~/datasets/wmt14/processed/plain/train.de").expanduser()
    valid_src_path=Path("~/datasets/wmt14/processed/plain/valid.en").expanduser()
    valid_tgt_path=Path("~/datasets/wmt14/processed/plain/valid.de").expanduser()
    checkpoint_directory=(project_root/"checkpoints")
    history_checkpoint_directory=(checkpoint_directory/"history")
    last_checkpoint_path=(checkpoint_directory/"last.pt")
    best_checkpoint_path=(checkpoint_directory/"best.pt")

    resume_path=last_checkpoint_path
    tokenizer=WMT14Tokenizer(tokenizer_path)

    model_config=get_model_config(model_size,tokenizer.vocab_size)

    train_dataset=data.WMT14Dataset(train_src_path,train_tgt_path,tokenizer,
                                    max_examples,max_tokens,buffer_size,seed)
    valid_dataset=data.WMT14Dataset(valid_src_path,valid_tgt_path,tokenizer,
                                    valid_max_examples,max_tokens,0,seed)
    
    net=build_model(model_config,device)
    initialize_parameters(net)
    optimizer=optimize.optimizer(net)

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
        start_epoch=saved_epoch+1
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
                                               training_history)
        epoch_losses.append(epoch_loss)
        
        validation_nll,_,_=evaluate_nll(net,valid_dataset,device,tokenizer.pad_id,
                                        token_budget,pool_size,seed)
        validation_nlls.append(validation_nll)

        training_history={"epoch_losses":epoch_losses,
                          "validation_nlls":validation_nlls}
        
        is_best=validation_nll<best_validation_nll
        if is_best:
            best_validation_nll=validation_nll
            checkpoint.save_checkpoint(best_checkpoint_path,net,optimizer,epoch_idx,
                                       global_step,best_validation_nll,model_config,
                                       training_history)

        checkpoint.save_checkpoint(last_checkpoint_path,net,optimizer,epoch_idx,
                                   global_step,best_validation_nll,model_config,
                                   training_history)

        history_checkpoint_path=(history_checkpoint_directory/(
            f"checkpoint_epoch_{epoch_idx + 1:04d}"f"_step_{global_step:08d}.pt"))
        checkpoint.save_model_checkpoint(history_checkpoint_path,net,epoch_idx,
                                         global_step,validation_nll,model_config)
        epoch_idx+=1

if __name__=="__main__":
    main()