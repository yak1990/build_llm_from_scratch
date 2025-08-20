from functools import partial
import json
import toy_qwen
from torch.utils.data import Dataset,DataLoader
import torch
import math
from torch.amp import autocast,GradScaler
from torch.optim.lr_scheduler import LambdaLR
import random
import numpy as np


def read_all_data(f_path):
    with open(f_path,'r') as f:
        data=json.load(f)
    return data

def formart_instruct_data(input_data):
    instruction=input_data['instruction'].strip()
    input=input_data['input'].strip()
    output=input_data['output'].strip()
    instruction = (
        f'<|im_start|>user\n'
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f'<|im_end|>\n'
        f'<|im_start|>Instruction\n'
        f"{instruction}"
        f'<|im_end|>\n'
    )

    input = (
                f"<|im_start|>Input:\n"
                f"{input}"
                f"<|im_end|>\n"
            ) if input else ""

    output_text = (
                    f"<|im_start|>response:\n"
                    f"{output}"
                    f"<|im_end|>\n"
                  )

    input_text=instruction+input

    return input_text,output_text



class InstructDataset(Dataset):
    def __init__(self,data_list,tokenizer) -> None:
        super().__init__()
        self.tokenizer=tokenizer

        input_list=[]
        output_list=[]
        for i in data_list:
            input_text,output_text=formart_instruct_data(i)
            input_list.append(tokenizer.encode(input_text))
            output_list.append(tokenizer.encode(output_text)+[tokenizer.eos_token_id])

        self.input_list=input_list
        self.output_list=output_list
        
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, index):
        return self.input_list[index],self.output_list[index]

def collate_fn(batch,padding_token_id,ignore_id):
    max_len=max([len(i[0])+len(i[1]) for i in batch])

    input_list=[]
    target_list=[]
    for i in batch:
        # input_len=len(i[0])
        # output_len=len(i[1])

        # now_input=i[0]+i[1]+[padding_token_id]*(max_len-input_len-output_len)
        # now_output=[ignore_id]*(input_len-1)+i[1]+[ignore_id]*(max_len+1-input_len-output_len)

        full_ids=i[0]+i[1]
        full_len=len(full_ids)
        now_input=full_ids+[padding_token_id]*(max_len-full_len)
        now_output=full_ids[1:]+[ignore_id]*(max_len+1-full_len)

        input_list.append(now_input)
        target_list.append(now_output)
    
    input_list=torch.tensor(input_list)
    target_list=torch.tensor(target_list)
    return input_list,target_list


def get_data(f_path,batch_size,shuffle=True,drop_last=True,num_workers=0,train_ratio=0.9):
    data=read_all_data(f_path)
    train_len=int(train_ratio*len(data))
    train_data=data[:train_len]
    eval_data=data[train_len:]

    tokenizer=toy_qwen.QwenTokenizer(r'Qwen3-0.6B/tokenizer.json')

    train_dataset=InstructDataset(train_data,tokenizer)
    eval_dataset=InstructDataset(eval_data,tokenizer)

    now_collate_fn=partial(collate_fn,padding_token_id=tokenizer.pad_token_id,ignore_id=-100)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers,collate_fn=now_collate_fn)
    eval_dataloader=DataLoader(eval_dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers,collate_fn=now_collate_fn)
    return tokenizer,train_dataset,eval_dataset,train_dataloader,eval_dataloader


def lr_lambda(step,init_lr,lr,min_lr,warm_up_num,train_count_num):
    run_count=step
    if run_count<warm_up_num:
        init_w=1-run_count/warm_up_num
        now_lr=init_lr*init_w+lr*(1-init_w)
    elif run_count<train_count_num*0.75:
        now_lr=lr
    else:
        progress=(run_count-warm_up_num)/(train_count_num-warm_up_num)
        now_lr=min_lr+(lr-min_lr)*0.5*(1+math.cos(progress*math.pi))
    return now_lr

def train():
    f_path=r'RawTextData/instruction_data.json'
    batch_size=8
    vocab_size=151936

    epochs=20

    # 加载数据集，分为训练集、测试集
    # 创建dataloader
    tokenizer,train_dataset,eval_dataset,train_loader,eval_loader=get_data(f_path,batch_size)

    max_run_count=epochs*len(train_loader)

    # # 设置网络
    model=toy_qwen.get_qwen()
    # model=torch.compile(model)
    

    # copy from gpt5
    device='cuda' if torch.cuda.is_available() else 'cpu'
    use_bf16=getattr(torch.cuda,'is_bf16_supported',lambda:False)()
    dtype=torch.bfloat16 if (device=='cpu' or use_bf16) else torch.float16


    # 设置优化器
    init_lr=1e-5
    lr=1e-4
    min_lr=1e-6
    weight_decay=1e-4
    warm_up_ratio=0.1
    optimizer=torch.optim.AdamW(model.parameters(),lr=init_lr,weight_decay=weight_decay)
    
    # # 优化时，使用warm up与cos decay
    # lr_func=partial(lr_lambda,init_lr=init_lr,lr=lr,min_lr=min_lr,warm_up_num=warm_up_ratio*max_run_count,train_count_num=max_run_count)
    # scheduler=LambdaLR(optimizer,lr_lambda=lr_func)

    scaler=GradScaler(enabled=(device=='cuda'))


    def train_step(x,y):
        optimizer.zero_grad()
        with autocast(device_type=device,dtype=dtype):
            outputs=model(x)
            loss=torch.nn.functional.cross_entropy(outputs.view(-1,vocab_size),y.view(-1))
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return loss.item()


    def eval_step(x,y):
        with autocast(device_type=device,dtype=dtype):
            outputs=model(x)
            loss=torch.nn.functional.cross_entropy(outputs.view(-1,vocab_size),y.view(-1))
            # predict=outputs.argmax(dim=-1,keepdim=True)
            # acc=(predict.view(-1)==y.view(-1)).float().mean()
            # return loss.item(),acc.item()
            return loss.item()

    def handle_batch(batch):
        x,y=batch
        x=x.to(device=device)
        y=y.to(device=device)
        return x,y

    def test_generate():
        model.eval()
        with torch.no_grad():
            random_id=random.choice([i for i in range(len(eval_dataset))])
            x,y=eval_dataset[random_id]

            input_text=tokenizer.decode(x)
            target_text=tokenizer.decode([j for j in y if j>=0])

            print(f'input_text : {input_text}')
            print(f'target_text : {target_text}')
            print(f'model output:')
            input_ids=torch.Tensor(x).int().cuda().unsqueeze(0)
            for token in toy_qwen.generate_text_stream(model,input_ids,max_new_tokens=100,eos_token_id=tokenizer.eos_token_id,top_k=50,temperature=1.2):
                token_id=token.squeeze(0).tolist()
                print(tokenizer.decode(token_id),end='',flush=True)
            print()
           


    # test_generate()

    # 训练
    for epoch in range(epochs):
        model.train()
        epoch_train=[]
        epoch_eval=[]

        # run train
        for batch in train_loader:
            now_loss=train_step(*handle_batch(batch))
            epoch_train.append(now_loss)
            # print(now_loss)
            # scheduler.step()
        

        # run eval
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                x,y=handle_batch(batch)
                now_loss=eval_step(x,y)
                epoch_eval.append(now_loss)

        # print(f'epoch:{epoch},train loss:{eval_model(model,train_loader)}')
        print(f'epoch:{epoch},train loss:{np.mean(epoch_train)}')
        print(f'epoch:{epoch},eval loss:{np.mean(epoch_eval)}')
        # 生成文本
        # print(f'{epoch} generate')
        if epoch>0 and epoch%5==0:
            test_generate()

    # print(his_lr)
    # import matplotlib.pyplot as plt
    # plt.plot(his_lr)
    # plt.show()

    # 获取eval数据
    # 获取random generate text


    # 保存模型参数，模型与优化器


def main():
    f_path=r'RawTextData/instruction_data.json'
    tokenizer,_,_,train_loader,eval_loader=get_data(f_path,batch_size=8)
    for id,i in enumerate(eval_loader):
        print('-'*100)
        print(id)
        input_list,target_list=i

        for j_id in range(len(input_list)):
            input_str=tokenizer.decode(input_list[j_id].tolist())
            output_str=tokenizer.decode([j for j in target_list[j_id].tolist() if j>=0])
            print('input_str\n',input_str)
            print('output_str\n',output_str)

        print(input_list.shape)
        print(target_list.shape)
        if id>1:
            break
   
if __name__=='__main__':
    # main()
    train()