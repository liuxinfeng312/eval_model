import csv
input_file = '2_kyc2_2023-10-17.csv'

with open(input_file,encoding='utf-8') as f:
    reader = csv.reader(f)
    header =next(reader)
    res_dict =dict()
    for line in reader:
        #print(line)
        k = line[0]
        if 'alpaca' in k:
            k = 'alpaca'
        if 'FinCUGE' in k:
            k='FinCUGE'
        if 'wyd' in k:
            k ='wyd'
        if 'wld' in k:
            k = 'wld'
        if 'chineseSquad' in k or 'cmrc'in k or 'dureader' in k:
            k='中文阅读理解'
            
        bleu =line[4]
        if k =='':
            continue
        if bleu == '':
            continue
            
        if k not in res_dict:
            res_dict[k] =[]
        res_dict[k] .append(round(float(bleu),4))
    
    with open('3_kyc2_subject.csv','w',encoding='utf-8',newline ='') as wf:
        writer = csv.writer(wf)
        header = ['file','条数','平均 bleu']
        writer.writerow(header)
        for k,v in res_dict.items():
            writer.writerow([k,len(v),sum(v)/len(v)])
     
