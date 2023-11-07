import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import csv 
from nltk.translate.bleu_score import sentence_bleu
import time


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
def load_model():
    
    tokenizer = AutoTokenizer.from_pretrained(r'/data/app/xfliu/Baichuan2-7B-Chat', use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(r'/data/app/xfliu/Baichuan2-7B-Chat', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(r'/data/app/xfliu/Baichuan2-7B-Chat')

    
    return model,tokenizer
    

def get_data():
    pass
    



def main():
    start=time.time()
    
    model,tokenizer = load_model()
    end1 = time.time()
    print('load model speed {}s'.format(str(end1-start)))
    
    
    #response, history = model.chat(tokenizer, "你好", history=[])
    #print(response)
    
    
    with open('1_baichuan2-7B.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        header = ['filename','instruction','output','model_ouput']
        writer.writerow(header)
        err_num=0
        all_num =0
        with open('../test2800_new.txt',encoding='utf-8') as reader:
            for li in tqdm(reader):
                all_num+=1
                try:
                    li_split = li.split('\t')
                    filename = li_split[0]
                    assert len(li_split)==2
                    li_json = json.loads(li_split[1])
                    instruction = li_json['instruction']+li_json['input']
                    messages = []
                    messages.append({"role": "user", "content": instruction})
                    response = model.chat(tokenizer, messages)
                    

                    writer.writerow([filename,instruction,li_json['output'],response])
                    f.flush()
                
                except Exception as e:
                    #print(li_split)
                    err_num+=1
                    print(e)
        print('总条数：',all_num,'错误跳过的条数：',err_num)
    end2 = time.time()
    print('all speed ',end2-start)

    
    
    
    #response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    #print(response)
   # pass
    
main()

