import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import csv 
from nltk.translate.bleu_score import sentence_bleu
import time
import fire 

def cer(str1, str2):
    # 初始化编辑距离矩阵
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    for i in range(len(str1) + 1):
        dp[i][0] = i
    for j in range(len(str2) + 1):
        dp[0][j] = j

    # 计算编辑距离
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

    # 计算CER
    num_errors = dp[-1][-1]
    cer = num_errors / max(len(str1), len(str2))
    return cer


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model = model.eval()
    return model,tokenizer
    

def get_data():
    pass
    



def main(model_path):
    start=time.time()
    
    model,tokenizer = load_model(model_path)
    end1 = time.time()
    print('load model speed {}s'.format(str(end1-start)))
    
    
    #response, history = model.chat(tokenizer, "你好", history=[])
    #print(response)
    
    output_name = os.path.basename(model_path)
    with open('1_{}.csv'.format(output_name),'w',encoding='utf-8',newline='') as f:
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
                    response, history = model.chat(tokenizer, instruction, history=[])
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
fire.Fire(main)    


