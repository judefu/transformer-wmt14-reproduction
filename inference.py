import torch
from torch.nn import functional as F

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
        for _ in range(max_new_tokens+len(src_ids)):
            logits=net.decoder(generated,state)
            next_token_logits=logits[:,-1,:]
            next_token_logits[:, tokenizer.pad_id] = float("-inf")
            next_token_logits[:, tokenizer.bos_id] = float("-inf")
            next_token=next_token_logits.argmax(dim=-1)[None,:]
            generated=torch.cat([generated,next_token],dim=1)
            if next_token.item()==tokenizer.eos_id:
                break
        generated_ids=generated[0].cpu().tolist()
        generated_ids=[token_id for token_id in generated_ids 
                       if token_id not in special_ids]
        ret=tokenizer.decode(generated_ids)
    return ret

def length_penalty(len, alpha=0.6):
    return ((5.0+len)/6.0)**alpha

def log_prob_scores(log_prob,len,alpha=0.6):
    len=max(len-1,1)
    return log_prob/length_penalty(len,alpha)

def beam_search(net,tokenizer,src_text,device,beam_size=4,max_new_tokens=100,
                alpha=0.6):
    net.eval()
    with torch.no_grad():
        src_ids=tokenizer.encode(src_text,add_eos=True)
        src=torch.tensor([src_ids],dtype=torch.long,device=device)
        src_valid_lens=torch.tensor([len(src_ids)],dtype=torch.long,device=device)
        enc_outputs=net.encoder(src,src_valid_lens)
        state=net.decoder.init_state(enc_outputs,src_valid_lens)
        beams=[{"tokens":[tokenizer.bos_id],"log_prob":0.0,"is_finished":False}]

        for _ in range(len(src_ids)+max_new_tokens):
            candidates=[]
            for beam in beams:
                if beam["is_finished"]:
                    candidates.append(beam)
                    continue
                dec_input=torch.tensor([beam["tokens"]],dtype=torch.long,
                                       device=device)
                logits=net.decoder(dec_input,state)
                log_probs=F.log_softmax(logits[0,-1,:],dim=-1)
                log_probs[tokenizer.pad_id]=float("-inf")
                log_probs[tokenizer.bos_id]=float("-inf")
                beam_log_probs,beam_tokens=torch.topk(log_probs,beam_size)

                for beam_log_prob,beam_token in zip(beam_log_probs.tolist(),
                                                    beam_tokens.tolist()):
                    candidates.append({"tokens":(beam["tokens"]+[beam_token]),
                                      "log_prob":(beam["log_prob"]+beam_log_prob),
                                      "is_finished":beam_token==tokenizer.eos_id})
            candidates=sorted(candidates,key=lambda candidates:log_prob_scores(
                candidates["log_prob"],len(candidates["tokens"]),alpha),reverse=True)
            beams=candidates[:beam_size]

            if all(beam["is_finished"] for beam in beams):
                break

        best_beam=None
        for beam in beams:
            if beam["is_finished"]:
                best_beam=beam
                break
        if best_beam is None:
            best_beam=beams[0]

        tokens=[token for token in best_beam["tokens"]
                if token not in{tokenizer.bos_id,tokenizer.eos_id,tokenizer.pad_id}]
        return tokenizer.decode(tokens)