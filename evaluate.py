import torch
import math
import loss
from sacrebleu.metrics import BLEU
from pathlib import Path
from inference import beam_search, greedy_decode
from tokenizer import WMT14Tokenizer
from transformer import Transformer
import data

def load_net(checkpoint_path,tokenizer_path,device):
    tokenizer=WMT14Tokenizer(tokenizer_path)
    checkpoint_data=torch.load(checkpoint_path,map_location="cpu",
                                 weights_only=False)
    
    model_config=dict(checkpoint_data["model_config"])
    net=Transformer(**model_config)
    net.load_state_dict(checkpoint_data["net_state_dict"])
    net=net.to(device)
    return net,tokenizer


def translate_file(net,tokenizer,src_path,ref_path,device,method,max_examples,
                   max_new_tokens,beam_size,alpha):
    srcs=[]
    hypos=[]
    refs=[]
    with (src_path.open("r", encoding="utf-8") as src_file,
          ref_path.open("r", encoding="utf-8") as ref_file):
        for src_line, ref_line in zip(src_file,ref_file,strict=True):
            if max_examples is not None:
                if len(hypos)>=max_examples:
                    break

            src_text=src_line.rstrip("\r\n")
            ref_text=ref_line.rstrip("\r\n")

            if method=="greedy":
                hypo=greedy_decode(net,tokenizer,src_text,device,max_new_tokens)
            else:
                hypo=beam_search(net,tokenizer,src_text,device,beam_size,
                                       max_new_tokens,alpha)
                
            srcs.append(src_text)
            hypos.append(hypo)
            refs.append(ref_text)
    return srcs, hypos, refs


def calculate_token_accuracy(logits,labels,pad_id):
    predictions=logits.argmax(dim=-1)
    mask= labels!=pad_id
    correct=((predictions==labels)&mask).sum().item()
    total=mask.sum().item()
    return correct, total

def save_hypos(hypos,output_path):
    output_path.parent.mkdir(parents=True,exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for hypo in hypos:
            output_file.write(hypo+"\n")


def calculate_bleu(hypos,refs):
    bleu=BLEU(tokenize="13a")
    result=bleu.corpus_score(hypos,[refs])
    signature=bleu.get_signature()
    return result, signature


def evaluate_nll(net,dataset,device,pad_id,max_tokens,pool_size,seed):
    net.eval()
    dataset.set_epoch(0)

    example_batches=data.batch_by_lens(iter(dataset),max_tokens,pool_size,seed)

    total_nll_sum=0.0
    total_correct=0
    total_tokens=0
    batch_count=0

    with torch.no_grad():
        for examples in example_batches:
            (src_batch,src_valid_lens,decoder_batch,_,label_batch
            )=data.collate_batch(examples,pad_id=pad_id)

            src_batch=src_batch.to(device)
            src_valid_lens=src_valid_lens.to(device)
            decoder_batch=decoder_batch.to(device)
            label_batch=label_batch.to(device)

            logits=net(src_batch,decoder_batch,src_valid_lens)
            batch_nll=loss.label_smooth_cross_entropy(logits,label_batch,pad_id,0.0)

            correct, token_count=calculate_token_accuracy(logits,label_batch,pad_id)

            total_nll_sum+=(batch_nll.item()*token_count)
            total_correct+=correct
            total_tokens+=token_count
            batch_count+=1

    average_nll=total_nll_sum/total_tokens
    ppl=math.exp(average_nll)
    token_acc=total_correct/total_tokens

    return average_nll,ppl,token_acc

def main(nll=False):
    project_directory=Path(__file__).resolve().parent
    data_directory=(Path.home()/"datasets"/"wmt14"/"processed"/"plain")
    tokenizer_path=(project_directory/"vocab"/"wmt14_en_de_bpe_37k.model")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if nll:
        seed=42
        valid_max_examples=500
        max_tokens=128
        token_budget=512
        pool_size=200

        valid_src_path=(data_directory/"valid.en")
        valid_tgt_path=(data_directory/"valid.de")
        checkpoint_path=(project_directory/"checkpoints"/"averaged_last_2.pt")

        net, tokenizer=load_net(checkpoint_path,tokenizer_path,device)
        valid_dataset=data.WMT14Dataset(valid_src_path,valid_tgt_path,tokenizer,
                                        valid_max_examples,max_tokens,0,seed)
        valid_nll,valid_ppl,valid_acc=evaluate_nll(net,valid_dataset,device,
                                                   tokenizer.pad_id,token_budget,
                                                   pool_size,seed)
        return  valid_nll,valid_ppl,valid_acc

    else:
        checkpoint_path=(project_directory/"checkpoints"/"best.pt")
        src_path=(data_directory/"valid.en")
        ref_path=(data_directory/"valid.de")

        max_examples=20
        method="greedy"
        max_new_tokens=50
        beam_size=4
        alpha=0.6
        output_path=(project_directory/"outputs"/f"valid.{method}.{max_examples}.de")

        net, tokenizer=load_net(checkpoint_path,tokenizer_path,device)
        net.eval()
        _, hypos, refs=translate_file(net,tokenizer,src_path,ref_path,device,method,
                                     max_examples,max_new_tokens,beam_size,alpha)
        save_hypos(hypos,output_path)

        bleu_result, bleu_signature=calculate_bleu(hypos,refs)

        print("\nnumber of evaluated sentences:", len(hypos))
        print("BLEU result:")
        print(bleu_result)
        print("\nSacreBLEU signature:")
        print(bleu_signature)

if __name__=="__main__":
    main()