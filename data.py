import torch
from tokenizer import WMT14Tokenizer
from pathlib import Path
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from functools import partial

def encode_pair(src_text,tgt_text,tokenizer):
    src_tokens=tokenizer.encode(src_text,add_eos=True)
    decoder_input=tokenizer.encode(tgt_text,add_bos=True)
    label=tokenizer.encode(tgt_text,add_eos=True)
    return src_tokens,decoder_input,label

def pad_lines(lines,pad_id=0):
    batch_size=len(lines)
    max_lenth=max(len(line) for line in lines)
    batch=torch.full((batch_size,max_lenth),fill_value=pad_id,dtype=torch.long)
    valid_lens=torch.tensor([len(line) for line in lines],dtype=torch.long)

    for i,line in enumerate(lines):
        batch[i,:len(line)]=torch.tensor(line,dtype=torch.long)

    return batch,valid_lens

def collate_batch(corpus,pad_id=0):
    src_corpus=[line[0] for line in corpus]
    tgt_corpus=[line[1] for line in corpus]
    label_corpus=[line[2] for line in corpus]

    src_batch,src_valid_lens=pad_lines(src_corpus,pad_id)
    tgt_batch,tgt_valid_lens=pad_lines(tgt_corpus,pad_id)
    label_batch,_=pad_lines(label_corpus,pad_id)

    return(src_batch,src_valid_lens,tgt_batch,tgt_valid_lens,label_batch)

class WMT14Dataset(IterableDataset):
    def __init__(self,src_path,tgt_path,tokenizer,maxlen=None):
        super().__init__()
        self.src_path=Path(src_path).expanduser()
        self.tgt_path=Path(tgt_path).expanduser()
        self.tokenizer=tokenizer
        self.maxlen=maxlen

    def __iter__(self):
        cnt=0
        with (self.src_path.open("r",encoding="utf-8") as src_file,
              self.tgt_path.open("r",encoding="utf-8") as tgt_file):
            for src_text,tgt_text in zip(src_file,tgt_file):
                if cnt==self.maxlen:
                    break
                if self.maxlen is not None:
                    cnt+=1

                src_text=src_text.rstrip("\n")
                tgt_text=tgt_text.rstrip("\n")
                yield encode_pair(src_text,tgt_text,self.tokenizer)