#! /usr/bin/env python
# -*- coding: utf-8 -*-
############################################################
#
# Copyright (c) 2023 WeBank Inc. Ltd. All Rights Reserved.
#
# File: test_chatglm2_chat.py
# Author: kinvapeng@webank.com
# Date: 2023-09-11 14:07
#
############################################################

import sys
from transformers import AutoTokenizer, AutoModel



def main():
 
    tokenizer = AutoTokenizer.from_pretrained("/data1/muzhili/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("/data1/muzhili/chatglm2-6b", trust_remote_code=True, device='cuda')
    model = model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)



if __name__ == "__main__":
    main()
