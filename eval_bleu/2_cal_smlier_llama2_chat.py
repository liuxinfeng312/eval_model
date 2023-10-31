import os
from tqdm import tqdm
import re
import csv 
from nltk.translate.bleu_score import sentence_bleu


def cal_cer(str1, str2):
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
    cer = 1-(num_errors / max(len(str1), len(str2)))
    return cer,num_errors,max(len(str1), len(str2))
    


def remove_non_chinese(text):
    # 使用正则表达式找到非中文字符串
    non_chinese_pattern = re.compile(r'[^\u4e00-\u9fa50-9]')
    non_chinese_str = non_chinese_pattern.findall(text)
    
    # 使用正则表达式除去非中文字符串
    result = re.sub(non_chinese_pattern, '', text)
    
    return result
    
    
def main():
    with open('2_bleu_llama2_chat.csv','w',encoding='utf-8',newline='') as wf:
        writer=csv.writer(wf)
        all_err =0
        all_len_num=0
        bleu_list = []
        with open('./1_res_chat.csv',encoding='utf-8') as f:
            reader = csv.reader(f)
            header =next(reader)+['bleu']
            writer.writerow(header)
            try:
                for line in tqdm(reader):
                    try:
                        output = line[2].replace('，','').replace('。','').replace('？','').replace('；','').replace('！','').replace('（','').replace('）','').replace('\n','')
                        if output =='':
                            writer.writerow(line+[''])
                            continue
                        
                        model_output = line[3].replace('，','').replace('。','').replace('？','').replace('；','').replace('！','').replace('（','').replace('）','').replace('\n','')
                        #cer,err,len_max = cal_cer(output,model_output)
                        #all_err+=err
                        #all_len_num+=len_max
                        
                        model_output_split = model_output.split('Assistant:',1)
                        #print(len(model_output_split))
                        assert len(model_output_split)==2
                        print(model_output)
                        print(model_output_split)
                        print('-'*20)
                        bleu = sentence_bleu([list(output)], list(model_output_split[1]))
                        bleu_list.append(round(float(bleu),4))
                        #print(output)
                        #print('-'*100)
                        #print(model_output)
                        #print('cer:',cer)
                        #print('='*100)
                        writer.writerow(line+[round(float(bleu),4)])
                        wf.flush()
                    except Exception as e:
                        print('err1:',e)
            except Exception as e2:
                print('err2:',e2)
        writer.writerow(['','','','','平均bleu:',sum(bleu_list)/len(bleu_list)])
        

main()
