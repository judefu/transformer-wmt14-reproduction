import loss
import optimize
import data

def calculate_token_accuracy(logits,labels,pad_id):
    predictions=logits.argmax(dim=-1)
    mask= labels!=pad_id
    correct=((predictions==labels)&mask).sum().item()
    total = mask.sum().item()
    return correct, total


def train_one_epoch(net,dataset,optimizer,device,pad_id,epoch_index,global_step,
                    num_hiddens,warmup_steps,max_tokens,pool_size,
                    seed):
    net.train()
    dataset.set_epoch(epoch_index)

    example_batches=data.batch_by_lens(iter(dataset),max_tokens,pool_size,
                                         seed+epoch_index)
    total_loss_sum=0.0
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
        optimize.update_learning_rate(optimizer,global_step,num_hiddens,
                                         warmup_steps)
        
        optimizer.step()

        correct,token_count=calculate_token_accuracy(logits,label_batch,pad_id)
        total_loss_sum+=(l.item()*token_count)
        total_correct+=correct
        total_tokens+=token_count

    average_loss=total_loss_sum/total_tokens
    average_accuracy=total_correct/total_tokens

    return (global_step,average_loss,average_accuracy)