import torch


class Beamobj(object):
        def __init__(self,tokens,log_prob,h,c):
                self.tokens=tokens
                self.log_prob=log_prob
                self.h=h
                self.c=c

        def get_token(self):
                return self.tokens[-1]
            
        def get_prob(self):
                return self.log_prob/len(self.tokens)
            
        def get_child(self,tokens,prob,h,c):
                return Beamobj(tokens,prob,h,c)
                
                
class BeamSearch(object):
        def __init__(self,word_to_int,int_to_word,max_decoding_len,min_len,beam_size):
                self.word_to_int=word_to_int
                self.int_to_word=int_to_word
                self.max_decode_len=max_decoding_len
                self.beam_size=beam_size
                self.min_len=min_len

        def beam_search(self,embeddings,encoder,decoder,dataloader):
                inp,inp_mask,lab,lab_mask=dataloader.load_next()
                inp_=embeddings(inp)
                enc_out,(h,c)=encoder(inp_,inp_mask)
                
                beam,results=[],[]
                beam.append(Beamobj([self.word_to_int['sos']],0,h,c))
                
                iters=0
                while iters<self.max_decode_len and len(results)<self.beam_size:
                        inps,hs,cs=[],[],[]
                        for instance in beam:
                                inps.append(instance.get_token())
                                hs.append(instance.h)
                                cs.append(instance.c)
                                
                        inps=torch.tensor(inps).view(len(beam),-1)
                        h=torch.cat(hs,dim=1)
                        c=torch.cat(cs,dim=1)
                        
                        dec_embeds=embeddings(inps)
                        enc_out_=enc_out.expand(dec_embeds.shape[0],-1,-1)
                        probs,(hs,cs)=decoder(dec_embeds,enc_out_,inp_mask,h,c)
                        probs,indi=torch.topk(probs.squeeze(1),self.beam_size,dim=-1)
                        
                        news=[]
                        for i,instance in enumerate(beam):
                                tokens=instance.tokens
                                pb=instance.log_prob
                                h_ins=hs[0][i].view(1,1,-1)
                                c_ins=cs[0][i].view(1,1,-1)
                                
                                for j in range(self.beam_size):
                                        news.append(instance. \
                                                    get_child(tokens+[indi[i][j].item()],
                                                              pb+probs[i][j].item(),h_ins,c_ins))
                                                    
                        beam=[]
                        news=sorted(news,key=lambda x:x.get_prob(),reverse=True)
                        for i,instance in enumerate(news):
                                if instance.get_token()==self.word_to_int['eos']:
                                        if len(instance.tokens)>self.min_len:
                                                results.append(instance)
                                else:
                                        beam.append(instance)
                                if len(beam)==self.beam_size or  \
                                len(results)==self.beam_size:
                                        break
                                    
                        iters+=1
                        
                if len(results)==0:
                        results=beam
                results=sorted(results,key=lambda x:x.get_prob(),reverse=True)
                ans=results[0].tokens
                return_ans,gold=[],[]
                for word in ans:
                        return_ans.append(self.int_to_word[word])
                for word in lab[0]:
                        gold.append(self.int_to_word[word.item()])
                        
                return return_ans,gold