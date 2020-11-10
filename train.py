import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.metrics import bleu_score

import argparse
import os

from utils import prepare_input,DataLoader,TestLoader
from model import Embeddings,Encoder,Decoder
from decode import BeamSearch

def train_iters(encoder,decoder,embeddings,trainloader,devloader,iters,
                lr,generator,max_norm):
    
        params=list(embeddings.parameters())+ \
        list(encoder.parameters())+list(decoder.parameters())
        optimizer=optim.Adam(params,lr=lr)
        
        scalar,bleu_=0,0
        for i in range(iters):
                encoder.train()
                decoder.train()
                embeddings.train()
                
                optimizer.zero_grad()
                loss=train_one_batch(encoder,decoder,
                                     embeddings,trainloader)
                scalar+=loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params,max_norm)
                optimizer.step()
                
                if (i+1)%100==0:
                        print('iteration={}, loss={}'.format(i+1,scalar/100))
                        scalar=0
                if (i+1)%1000==0:
                        encoder.eval()
                        decoder.eval()
                        embeddings.eval()
                    
                        bleu=evaluate(encoder,decoder,
                                      embeddings,devloader,generator)
                        if bleu>bleu_:
                                bleu_=bleu
                                torch.save({
                                        'encoder':encoder.state_dict(),
                                        'decoder':decoder.state_dict(),
                                        'embeddings':embeddings.state_dict()
                                        },'model.pth')
                        print('evaluation bleu score={}'.format(bleu))

def train_one_batch(encoder,decoder,embeddings,trainloader):
        inp,inp_mask,lab,lab_mask=trainloader.load_next()
        
        inp_embeds=embeddings(inp)
        enc_out,(h,c)=encoder(inp_embeds,inp_mask)
        
        loss=0
        for i in range(lab_mask.shape[-1]-1):
                dec_embeds=embeddings(lab[:][i].view(-1,1))
                logits,(h,c)=decoder(dec_embeds,enc_out,inp_mask,h,c)
                logits=logits.squeeze(1)
                
                lossFunction=nn.CrossEntropyLoss(ignore_index=0)
                loss+=lossFunction(logits,lab[:][i+1])
                
        return loss

def evaluate(encoder,decoder,embeddings,devloader,generator):
        gens,labels=[],[]
        for _ in range(devloader.len):
                out,gold=generator.beam_search(embeddings,encoder,decoder,devloader)
                gens.append(out)
                labels.append([gold])
                
        return 100*bleu_score(gens,labels)

def main(args):
    
        train,dev,word_to_int,int_to_word \
        =prepare_input(args.dev_com_path,args.train_com_path,
                       args.dev_cose_path,args.train_cose_path)
        
        trainloader=DataLoader(train,args.bs)
        devloader=TestLoader(dev)
        
        encoder=Encoder(args.input_size,args.hidden_size) #.cuda()
        decoder=Decoder(args.hidden_size,args.input_size,len(word_to_int)) #.cuda() 
        embeddings=Embeddings(args.input_size,len(word_to_int)) #.cuda()
        generator=BeamSearch(word_to_int,int_to_word,args.max_decode_len,
                             args.min_decode_len,args.beam_size)
        
        if args.use_pretrained:
                params=torch.load(args.pretrained_path)
                encoder.load_state_dict(params['encoder'])
                decoder.load_state_dict(params['decoder'])
                embeddings.load_state_dict(params['embeddings'])
        
        train_iters(encoder,decoder,embeddings,trainloader,
                    devloader,args.iters,args.lr,generator,args.max_norm)

def setup():
        parser=argparse.ArgumentParser()
        parser.add_argument('--lr',type=float,default=0.001)
        parser.add_argument('--input_size',type=int,default=128)
        parser.add_argument('--hidden_size',type=int,default=256)
        parser.add_argument('--dev_com_path',type=str,
                            default='/home/pranav/ml/T5/dev_rand_split.jsonl')
        parser.add_argument('--train_com_path',type=str,
                            default='/home/pranav/ml/T5/train_rand_split.jsonl')
        parser.add_argument('--dev_cose_path',type=str,
                            default='/home/pranav/ml/T5/cose_dev.jsonl')
        parser.add_argument('--train_cose_path',type=str,
                            default='/home/pranav/ml/T5/cose_train.jsonl')
        parser.add_argument('--use_pretrained',type=bool,default=False)
        parser.add_argument('--pretrained_path',type=str,default=os.getcwd()+'/model.pth')
        parser.add_argument('--iters',type=int,default=10000)
        parser.add_argument('--bs',type=int,default=16)
        parser.add_argument('--max_norm',type=float,default=0.5)
        parser.add_argument('--min_decode_len',type=int,default=12)
        parser.add_argument('--max_decode_len',type=int,default=20)
        parser.add_argument('--beam_size',type=int,default=4)
        
        args=parser.parse_args()
        
        return args
    
if __name__=='__main__':
        args=setup()
        main(args)