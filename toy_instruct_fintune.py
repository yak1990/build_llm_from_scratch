import json
import toy_qwen
from torch.utils.data import Dataset,DataLoader
import torch
import math

def read_all_data(f_path):
    with open(f_path,'r') as f:
        data=json.load(f)
    return data

def formart_instruct_data(input_data):
    instruction=input_data['instruction'].strip()
    input=input_data['input'].strip()
    output=input_data['output'].strip()
    instruction = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
    )

    input = f"\n\n### Input:\n{input}" if input else ""

    output_begin = f"\n\n### response:\n"

    input_text=instruction+input+output_begin
    output_text=output

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
        input_len=len(i[0])
        output_len=len(i[1])

        now_input=i[0]+i[1]+[padding_token_id]*(max_len-input_len-output_len)
        now_output=[ignore_id]*(input_len-1)+i[1]+[ignore_id]*(max_len+1-input_len-output_len)

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

    from functools import partial
    now_collate_fn=partial(collate_fn,padding_token_id=tokenizer.pad_token_id,ignore_id=-100)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers,collate_fn=now_collate_fn)
    eval_dataloader=DataLoader(eval_dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers,collate_fn=now_collate_fn)
    return tokenizer,train_dataloader,eval_dataloader


def train():
    f_path=r'RawTextData/instruction_data.json'
    batch_size=4
    vocab_size=151936

    # 加载数据集，分为训练集、测试集
    # 创建dataloader
    tokenizer,train_loader,eval_loader=get_data(f_path,batch_size)
    # # 设置网络
    model=toy_qwen.get_qwen()
    
    # 设置优化器
    init_lr=1e-5
    lr=1e-4
    min_lr=1e-6
    weight_decay=1e-4
    warm_up_ratio=0.1
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)

    # 训练
    # 优化时，使用warm up与cos decay
    epochs=10
    max_run_count=epochs*len(train_loader)
    warm_up_count=int(warm_up_ratio*max_run_count)
    run_count=0
    his_lr=[]
    for epoch in range(epochs):
        model.train()
        epoch_train=[]
        epoch_eval=[]

        # run train
        for batch in train_loader:
            x,y=batch
            x=x.cuda()
            y=y.cuda()
            optimizer.zero_grad()

            # update lr
            if run_count<warm_up_count:
                init_w=1-run_count/warm_up_count
                now_lr=init_lr*init_w+lr*(1-init_w)
            else:
                progress=(run_count-warm_up_count)/(max_run_count-warm_up_count)
                now_lr=min_lr+(lr-min_lr)*0.5*(1+math.cos(progress*math.pi))
            his_lr.append(now_lr)
            run_count+=1
            for param_group in optimizer.param_groups:
                param_group['lr']=now_lr
            
            outputs=model(x)
            loss=torch.nn.functional.cross_entropy(outputs.view(-1,vocab_size),y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_train.append(loss.item())
            print(loss.item())
        

        # run eval
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                x,y=batch
                x=x.cuda()
                y=y.cuda()
                
                outputs=model(x)
                loss=torch.nn.functional.cross_entropy(outputs.view(-1,vocab_size),y.view(-1))
               
                epoch_eval.append(loss.item())
        # print(f'epoch:{epoch},train loss:{eval_model(model,train_loader)}')
        print(f'epoch:{epoch},train loss:{len(epoch_train)}')
        print(f'epoch:{epoch},eval loss:{len(epoch_eval)}')
        # 生成文本
        print(f'{epoch} generate')

    # print(his_lr)
    # import matplotlib.pyplot as plt
    # plt.plot(his_lr)
    # plt.show()

    # 获取eval数据
    # 获取random generate text


    # 保存模型参数，模型与优化器


def main():
    f_path=r'RawTextData/instruction_data.json'
    tokenizer,train_loader,eval_loader=get_data(f_path,batch_size=8)
    for id,i in enumerate(train_loader):
        print('-'*100)
        print(id)
        input_list,target_list=i

        # for j_id in range(len(input_list)):
        #     input_str=tokenizer.decode(input_list[j_id].tolist())
        #     output_str=tokenizer.decode([j for j in target_list[j_id].tolist() if j>=0])
        #     print(input_str)
        #     print(output_str)

        print(input_list.shape)
        print(target_list.shape)
        # if id>3:
        #     break
   
if __name__=='__main__':
    # main()
    train()