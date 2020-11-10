import json
from nltk.tokenize import WordPunctTokenizer
import random

import torch

def prepare_input(dev_com_path,train_com_path,dev_cose_path,train_cose_path):
        tokenizer=WordPunctTokenizer()
        
        dev_com=list(open(dev_com_path))
        result_dev=[json.loads(jline) for jline in dev_com]
        train_com=list(open(train_com_path))
        result_train=[json.loads(jline) for jline in train_com]
        
        dev_cose=list(open(dev_cose_path))
        exp_dev=[json.loads(jline) for jline in dev_cose]
        train_cose=list(open(train_cose_path))
        exp_train=[json.loads(jline) for jline in train_cose]
        
        dict_train,dict_dev,dict_train_cose,dict_dev_cose={},{},{},{}
        for line in result_dev:
                dict_dev[line['id']]=line['question']['stem']  #+' </s> '+ \
                # line['question']['choices'][ord(line['answerKey'])-ord('A')]['text']
        for line in result_train:
                dict_train[line['id']]=line['question']['stem'] #+' </s> '+ \
                # line['question']['choices'][ord(line['answerKey'])-ord('A')]['text']
                
        for line in exp_dev:
                dict_dev_cose[line['id']]=line['explanation']['open-ended']
        for line in exp_train:
                dict_train_cose[line['id']]=line['explanation']['open-ended']
                
        train_data,dev_data=[],[]
        se=set()
        for k in dict_train:
                tkd=tokenizer.tokenize(dict_train[k])
                tkdy=tokenizer.tokenize(dict_train_cose[k])
                train_data.append([tkd,tkdy])
                for token in tkd:
                        se.add(token)
                for token in tkdy:
                        se.add(token)
                        
        for k in dict_dev:
                tkd=tokenizer.tokenize(dict_dev[k])
                tkdy=tokenizer.tokenize(dict_dev_cose[k])
                dev_data.append([tkd,tkdy])
                for token in tkd:
                        se.add(token)
                for token in tkdy:
                        se.add(token)

        word_to_int,int_to_word={},{}
        i=4
        word_to_int['padt'],word_to_int['sos'],word_to_int['eos'],word_to_int['unk']=0,1,2,3
        int_to_word[0],int_to_word[1],int_to_word[2],int_to_word[3]='padt','sos','eos','unk'
        for token in se:
                word_to_int[token]=i
                int_to_word[i]=token
                i+=1
                
        train,dev=[],[]
        for line in train_data:
                inp=line[0]
                lab=line[1]
                ip,lb=[word_to_int['sos']],[word_to_int['sos']]
                for word in inp:
                        ip.append(word_to_int[word])
                for word in lab:
                        lb.append(word_to_int[word])
                ip.append(word_to_int['eos'])
                lb.append(word_to_int['eos'])
                train.append([ip,lb])
                
        for line in dev_data:
                inp=line[0]
                lab=line[1]
                ip,lb=[word_to_int['sos']],[word_to_int['sos']]
                for word in inp:
                        ip.append(word_to_int[word])
                for word in lab:
                        lb.append(word_to_int[word])
                ip.append(word_to_int['eos'])
                lb.append(word_to_int['eos'])
                dev.append([ip,lb])
        
        return train,dev,word_to_int,int_to_word
    
    
class DataLoader(object):
        def __init__(self,data,bs):
                self.data=data
                self.bs=bs
                
        def _get_data(self):
                data=random.sample(self.data,self.bs)
                
                return data
            
        def load_next(self):
                data=self._get_data()
                
                max_len_inp,max_len_lab=0,0
                for line in data:
                    max_len_inp=max(max_len_inp,len(line[0]))
                    max_len_lab=max(max_len_lab,len(line[1]))
                    
                inp=torch.zeros(self.bs,max_len_inp,dtype=torch.long) #.cuda()
                lab=torch.zeros(self.bs,max_len_lab,dtype=torch.long) #.cuda()
                inp_mask=torch.zeros(self.bs,max_len_inp,dtype=torch.long) #.cuda()
                lab_mask=torch.zeros(self.bs,max_len_lab,dtype=torch.long) #.cuda()
                
                for i,line in enumerate(data):
                        inp[i][:len(line[0])]=torch.tensor(line[0])
                        lab[i][:len(line[1])]=torch.tensor(line[1])
                        inp_mask[i][:len(line[0])]=1
                        lab_mask[i][:len(line[1])]=1
                        
                return inp,inp_mask,lab,lab_mask
            
            
class TestLoader(DataLoader):
        def __init__(self,data):
                self.data=data
                self.bs=1
                self.counter=0
                self.len=len(data)
                
        def _get_data(self):
                data=[self.data[self.counter]]
                self.counter=(self.counter+1)%self.len
                
                return data