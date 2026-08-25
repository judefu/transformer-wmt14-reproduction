import torch

def greedy_decode(net,tokenizer,src_text,device,max_new_tokens):
    net.eval()
    special_ids={tokenizer.bos_id,tokenizer.eos_id,tokenizer.pad_id}
    with torch.no_grad():
        src_ids=tokenizer.encode(src_text,add_eos=True)
        src=torch.tensor([src_ids],dtype=torch.long,device=device)
        src_valid_lens=torch.tensor([len(src_ids)],dtype=torch.long,device=device)

        enc_outputs=net.encoder(src,src_valid_lens)
        state=net.decoder.init_state(enc_outputs,src_valid_lens)
        generated=torch.tensor([[tokenizer.bos_id]],dtype=torch.long,device=device)
        for _ in range(max_new_tokens):
            logits=net.decoder(generated,state)
            next_token_logits=logits[:,-1,:]
            next_token=next_token_logits.argmax(dim=-1)[None,:]
            generated=torch.cat([generated,next_token],dim=1)
            if next_token.item()==tokenizer.eos_id:
                break
        generated_ids=generated[0].cpu().tolist()
        generated_ids=[token_id for token_id in generated_ids 
                       if token_id not in special_ids]
        ret=tokenizer.decode(generated_ids)
    return ret

def length_penalty(length, alpha):
    return ((5.0+length)/6.0)**alpha