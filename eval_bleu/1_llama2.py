import torch
from peft import PeftModel
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import csv 
from nltk.translate.bleu_score import sentence_bleu
import time


max_new_tokens = 2048
generation_config = dict(
            temperature=0.9,
                top_k=30,
                    top_p=0.6,
                        do_sample=True,
                            num_beams=1,
                                repetition_penalty=1.2,
                                    max_new_tokens=max_new_tokens
                                    )
def load_model():
    load_type = torch.float16 #Sometimes may need torch.float32
    device = torch.device(0)
#   if torch.cuda.is_available():
  #      device = torch.device(0)
   # else:
    #    device = torch.device('cpu')

   
    #tokenizer = LlamaTokenizer.from_pretrained("/data/app/user_data/muzhili/llama2/Llama-2-7b-hf")
  
    tokenizer = AutoTokenizer.from_pretrained("/data/app/user_data/muzhili/llama2/Llama-2-7b-hf")
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained("/data/app/user_data/muzhili/llama2/Llama-2-7b-hf")

    
    model = AutoModelForCausalLM.from_pretrained("/data/app/user_data/muzhili/llama2/Llama-2-7b-hf", torch_dtype=load_type, config=model_config, device_map='auto')

    #if device==torch.device('cpu'):
     #   model.float()

    model.eval()
    print("Load model successfully")
   
    return model,tokenizer,device
    

def get_data():
    pass
    



def main():
    start=time.time()
    
    model,tokenizer,device = load_model()
    end1 = time.time()
    print('load model speed {}s'.format(str(end1-start)))
    
    
    #response, history = model.chat(tokenizer, "你好", history=[])
    #print(response)
    
    
    with open('1_res.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        header = ['filename','instruction','output','model_ouput']
        writer.writerow(header)
        err_num=0
        all_num =0
        with open('./test2800_new.txt',encoding='utf-8') as reader:
            for li in tqdm(reader):
                all_num+=1
                try:
                    li_split = li.split('\t')
                    filename = li_split[0]
                    assert len(li_split)==2
                    li_json = json.loads(li_split[1])
                    instruction = li_json['instruction']+li_json['input']
                    
                    instruction="Human: \n" + instruction+ "\n\nAssistant:\n"
                    
                    inputs = tokenizer(
                        instruction,
                        add_special_tokens=False,
                        return_tensors="pt"
                            )
                    generation_output = model.generate(
                    input_ids = inputs["input_ids"].to(device), 
                                **generation_config
                                )[0]

                    generate_text = tokenizer.decode(       generation_output,skip_special_tokens=True)
                    
                    
                    print(instruction)
                    print('-'*10)
                    print(generate_text)
                    
                    writer.writerow([filename,instruction,li_json['output'],generate_text])
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

