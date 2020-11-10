import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class Embeddings(nn.Module):
        def __init__(self,embed_size,vocab_size):
                super(Embeddings,self).__init__()
                self.embeddings=nn.Embedding(vocab_size,embed_size,padding_idx=0)
                
        def forward(self,input):
                output=self.embeddings(input)
                return output
            

class Encoder(nn.Module):
        def __init__(self,input_size,hidden_size):
                super(Encoder,self).__init__()
                self.lstm=nn.LSTM(input_size,hidden_size,bidirectional=True,batch_first=True)
                self.wh=nn.Linear(2*hidden_size,hidden_size)
                
                self.reducer=nn.Linear(hidden_size*2,hidden_size)
                self.dropout=nn.Dropout(0.4)
                
        def forward(self,input,input_mask):
                inp_mask=torch.sum(input_mask,dim=-1)  
                packed_seq=pack_padded_sequence \
                (input,inp_mask,batch_first=True,enforce_sorted=False)
                output,(hidden,cell_state)=self.lstm(packed_seq)
                output,_=pad_packed_sequence(output,batch_first=True)
                output=self.dropout(self.wh(output))
                
                hidden=hidden.transpose(0,1).contiguous().view(output.shape[0],1,-1)
                cell_state=cell_state.transpose(0,1).contiguous().view(output.shape[0],1,-1)
                hidden=self.reducer(hidden)
                cell_state=self.reducer(cell_state)
                
                return output,(hidden,cell_state)
            
            
class Decoder(nn.Module):
        def __init__(self,hidden_size,input_size,vocab_size):
                super(Decoder,self).__init__()
                self.lstm=nn.LSTM(input_size,hidden_size)
                
                self.out1=nn.Linear(hidden_size*2,hidden_size)
                self.out2=nn.Linear(hidden_size,vocab_size)
                
                self.dropout1=nn.Dropout(0.4)
                self.dropout2=nn.Dropout(0.4)
                
        def forward(self,input,enc_out,enc_mask,h,c):
                output,(h,c)=self.lstm(input,(h,c))
                output=self.dropout1(output)
                
                interact=torch.matmul(output,enc_out.transpose(-2,-1))
                mask=enc_mask.unsqueeze(1)
                interact=interact+(mask-1)*-1e9
                attn=F.softmax(interact,dim=-1)
                context=torch.matmul(attn,enc_out)
                
                out1=self.out1(torch.tanh(torch.cat([output,context],dim=-1)))
                out1=self.dropout2(out1)
                out2=self.out2(out1)
                
                return out2,(h,c)