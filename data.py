import torch
import random
from pathlib import Path
from torch.utils.data import IterableDataset

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
    def __init__(self,src_path,tgt_path,tokenizer,max_examples=None,
                 max_tokens=256,buffer_size=0,seed=42):
        super().__init__()
        self.src_path=Path(src_path).expanduser()
        self.tgt_path=Path(tgt_path).expanduser()
        self.tokenizer=tokenizer
        self.max_examples=max_examples
        self.max_tokens=max_tokens
        self.buffer_size=buffer_size
        self.seed=seed
        self.epoch=0

    def set_epoch(self,epoch):
        self.epoch=epoch

    def generate_examples(self):
        cnt=0
        with (self.src_path.open("r",encoding="utf-8") as src_file,
              self.tgt_path.open("r",encoding="utf-8") as tgt_file):
            for src_text,tgt_text in zip(src_file,tgt_file):            
                if cnt==self.max_examples:
                    break

                src_text=src_text.rstrip("\n")
                tgt_text=tgt_text.rstrip("\n")
                example=encode_pair(src_text,tgt_text,self.tokenizer)

                if (len(example[0]) > self.max_tokens or 
                    len(example[1]) > self.max_tokens):
                    continue
                if self.max_examples is not None:
                    cnt+=1
                yield example

    def __iter__(self):
        examples=self.generate_examples()

        if self.buffer_size>1:
            yield from buffered_shuffle(examples,self.buffer_size,
                                        self.seed+self.epoch)
        else:
            yield from examples


def buffered_shuffle(iterator, buffer_size, seed):
    random_generator = random.Random(seed)
    buffer = []

    for item in iterator:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue

        index = random_generator.randrange(buffer_size)

        yield buffer[index]
        buffer[index] = item

    random_generator.shuffle(buffer)
    yield from buffer

def batch_by_tokens(iterator,max_tokens_per_batch):
    batch=[]
    max_src_len,max_tgt_len=0,0
    for example in iterator:
        src,tgt,_=example
        new_batch_size=len(batch)+1
        new_src_len=max(max_src_len,len(src))
        new_tgt_len=max(max_tgt_len,len(tgt))
        new_src_tokens=new_batch_size*new_src_len
        new_tgt_tokens=new_batch_size*new_tgt_len

        is_overload=(new_src_tokens>max_tokens_per_batch or 
                     new_tgt_tokens>max_tokens_per_batch)
        if batch and is_overload:
            yield batch

            batch=[]
            max_src_len,max_tgt_len=0,0

        batch.append(example)
        max_src_len=max(max_src_len,len(src))
        max_tgt_len=max(max_tgt_len,len(tgt))
    if batch:
        yield batch

def get_lens(example):
    src,tgt,_=example
    return max(len(src),len(tgt))

def batch_by_lens(iterator,max_tokens_per_batch,pool_size,seed):
    random_genrator=random.Random(seed)
    pool=[]

    for example in iterator:
        pool.append(example)
        if len(pool)==pool_size:
            sorted_pool=sorted(pool,key=get_lens)
            batches=list(batch_by_tokens(iter(sorted_pool),max_tokens_per_batch))
            random_genrator.shuffle(batches)
            yield from batches
            pool=[]

    if pool:
        sorted_pool=sorted(pool,key=get_lens)
        batches=list(batch_by_tokens(iter(sorted_pool),max_tokens_per_batch))
        random_genrator.shuffle(batches)
        yield from batches