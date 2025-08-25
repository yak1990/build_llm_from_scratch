import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.amp import autocast,GradScaler
from torch.optim.lr_scheduler import LambdaLR

from functools import partial
import copy
import random

import toy_instruct_fintune
import toy_qwen
import numpy as np

def formart_instruct_perference_data(input_data):
    instruction=input_data['instruction'].strip()
    perfect=input_data['chosen'].strip()
    reject=input_data['rejected'].strip()
    input=input_data['input'].strip()

    instruction = (
        # f'<|im_start|>user\n'
        # f"Below is an instruction that describes a task. "
        # f"Write a response that appropriately completes the request."
        # f'<|im_end|>\n'
        f'<|im_start|>Instruction\n'
        f"{instruction}"
        f'<|im_end|>\n'
    )

    input = (
                f"<|im_start|>Input:\n"
                f"{input}"
                f"<|im_end|>\n"
            ) if input else ""
    prompt=instruction+input

    perfect_text=(
        f'{prompt}'
        f"<|im_start|>Response:\n"
        f"{perfect}"
        f"<|im_end|>"
    )
    reject_text=(
        f'{prompt}'
        f"<|im_start|>Response:\n"
        f"{reject}"
        f"<|im_end|>"
    )

    return prompt,perfect_text,reject_text


class InstructPerferenceDataset(Dataset):
    def __init__(self,data_list,tokenizer) -> None:
        super().__init__()
        self.tokenizer=tokenizer

        prompt_list=[]
        perfect_list=[]
        reject_list=[]
        for i in data_list:
            prompt,perfect_text,reject_text=formart_instruct_perference_data(i)
            prompt_list.append(tokenizer.encode(prompt))
            perfect_list.append(tokenizer.encode(perfect_text)+[tokenizer.eos_token_id])
            reject_list.append(tokenizer.encode(reject_text)+[tokenizer.eos_token_id])

        self.prompt_list=prompt_list
        self.perfect_list=perfect_list
        self.reject_list=reject_list
        
    def __len__(self):
        return len(self.prompt_list)
    
    def __getitem__(self, index):
        return self.prompt_list[index],self.perfect_list[index],self.reject_list[index]

def collate_fn(batch,padding_token_id):
    max_perfect_len=max([len(i[1]) for i in batch])
    max_reject_len=max([len(i[2]) for i in batch])

    out={}
    out['prompt']=[]
    out['perfect']=[]
    out['perfect_mask']=[]
    out['reject']=[]
    out['reject_mask']=[]

    for i in batch:
        now_prompt,now_perfect,now_reject=i

        prompt_len=len(now_prompt)
        perfect_len=len(now_perfect)
        reject_len=len(now_reject)
        perfect_data=now_perfect+[padding_token_id]*(max_perfect_len-len(now_perfect))
        reject_data=now_reject+[padding_token_id]*(max_reject_len-len(now_reject))

        front_mask=3
        end_mask=3
        perfect_mask=[1]*(prompt_len+front_mask)+[0]*(perfect_len-prompt_len-front_mask-end_mask)+[1]*(max_perfect_len-perfect_len+end_mask)

        reject_mask=[1]*(prompt_len+front_mask)+[0]*(reject_len-prompt_len-front_mask-end_mask)+[1]*(max_reject_len-reject_len+end_mask)

        out['prompt'].append(now_prompt)
        out['perfect'].append(perfect_data)
        out['perfect_mask'].append(perfect_mask)
        out['reject'].append(reject_data)
        out['reject_mask'].append(reject_mask)
    
    out['perfect']=torch.tensor(out['perfect'])
    out['perfect_mask']=torch.tensor(out['perfect_mask'])
    out['reject']=torch.tensor(out['reject'])
    out['reject_mask']=torch.tensor(out['reject_mask'])
    return out

def get_data(f_path,batch_size,shuffle=True,drop_last=True,num_workers=0,train_ratio=0.9):
    data=toy_instruct_fintune.read_all_data(f_path)
    train_len=int(train_ratio*len(data))
    train_data=data[:train_len]
    eval_data=data[train_len:]

    tokenizer=toy_qwen.QwenTokenizer(r'Qwen3-0.6B/tokenizer.json')

    train_dataset=InstructPerferenceDataset(train_data,tokenizer)
    eval_dataset=InstructPerferenceDataset(eval_data,tokenizer)

    now_collate_fn=partial(collate_fn,padding_token_id=tokenizer.pad_token_id)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers,collate_fn=now_collate_fn)
    eval_dataloader=DataLoader(eval_dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers,collate_fn=now_collate_fn)
    return tokenizer,train_dataset,eval_dataset,train_dataloader,eval_dataloader

def compute_log_prob(model,input_data,target_mask):
    output=model(input_data)[:,:-1,:]
    log_prob=torch.nn.functional.log_softmax(output,dim=-1)
    target=input_data[:,1:]
    target_mask=target_mask[:,1:]
    log_prob=log_prob.gather(dim=-1,index=target.unsqueeze(-1)).squeeze(-1)
    log_prob=log_prob*(1-target_mask)

    base_data=(1-target_mask).sum()
    log_prob=log_prob.sum()/base_data
    return log_prob

def train():
    import os

    out_dir='qwen_dpo'
    os.makedirs(out_dir,exist_ok=True)

    f_path=r'RawTextData/instruction_data_with_preference.json'
    batch_size=8
    vocab_size=151936

    epochs=20

    # 加载数据集，分为训练集、测试集
    # 创建dataloader
    tokenizer,train_dataset,eval_dataset,train_loader,eval_loader=get_data(f_path,batch_size)

    max_run_count=epochs*len(train_loader)

    # # 设置网络
    model=toy_qwen.get_qwen()
    sft_path=r'qwen_finetune/17_0.4794456735253334.pth'
    model.load_state_dict(torch.load(sft_path))
    ref_model=copy.deepcopy(model)
    ref_model=ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_=False
    # model=torch.compile(model)
    

    # copy from gpt5
    device='cuda' if torch.cuda.is_available() else 'cpu'
    use_bf16=getattr(torch.cuda,'is_bf16_supported',lambda:False)()
    dtype=torch.bfloat16 if (device=='cpu' or use_bf16) else torch.float16


    # 设置优化器
    init_lr=1e-7
    lr=1e-6
    min_lr=1e-7
    weight_decay=1e-2
    warm_up_ratio=0.1
    optimizer=torch.optim.AdamW(model.parameters(),lr=init_lr,weight_decay=weight_decay)
    
    # 优化时，使用warm up与cos decay
    lr_func=partial(toy_instruct_fintune.lr_lambda,init_lr=init_lr,lr=lr,min_lr=min_lr,warm_up_num=warm_up_ratio*max_run_count,train_count_num=max_run_count)
    scheduler=LambdaLR(optimizer,lr_lambda=lr_func)

    scaler=GradScaler(enabled=(device=='cuda'))

    beta=0.1
    # min_batch_count=5
    # eval_batch_count=int(len(train_loader)/min_batch_count)

    
    def run_once(model,ref_model,perfect_data,prefect_mask,reject_data,reject_mask):
        with autocast(device_type=device,dtype=dtype):
            with torch.no_grad():
                ref_perfect_log_prob=compute_log_prob(ref_model,perfect_data,prefect_mask)
                ref_reject_log_prob=compute_log_prob(ref_model,perfect_data,prefect_mask)
                ref_log_prob=ref_perfect_log_prob-ref_reject_log_prob
            perfect_log_prob=compute_log_prob(model,perfect_data,prefect_mask)
            reject_log_prob=compute_log_prob(model,reject_data,reject_mask)
            log_prob=perfect_log_prob-reject_log_prob

            diff_log_prob=beta*(log_prob-ref_log_prob)
            loss=-F.logsigmoid(diff_log_prob).mean()

            
            perfect_loss=(perfect_log_prob-ref_perfect_log_prob).mean()
            reject_loss=(reject_log_prob-ref_reject_log_prob).mean()

            return loss,perfect_loss,reject_loss

    def test_generate():
        model.eval()
        with torch.no_grad():
            random_id=random.choice([i for i in range(len(eval_dataset))])
            prompt,perfect_data,reject_data=eval_dataset[random_id]

            input_text=tokenizer.decode(prompt)
            perfect_text=tokenizer.decode([j for j in perfect_data if j>=0])
            reject_text=tokenizer.decode([j for j in reject_data if j>=0])

            print()
            print(f'perfect_text : {perfect_text}')
            print()
            print(f'reject_text : {reject_text}')
            print()
            print(f'prompt : {input_text}')
            print()
            print(f'model output:')
            input_ids=torch.Tensor(prompt).int().cuda().unsqueeze(0)
            for token in toy_qwen.generate_text_stream(model,input_ids,max_new_tokens=100,eos_token_id=tokenizer.eos_token_id):
                token_id=token.squeeze(0).tolist()
                print(tokenizer.decode(token_id),end='',flush=True)

            print(f'ref model output:')
            input_ids=torch.Tensor(prompt).int().cuda().unsqueeze(0)
            for token in toy_qwen.generate_text_stream(ref_model,input_ids,max_new_tokens=100,eos_token_id=tokenizer.eos_token_id):
                token_id=token.squeeze(0).tolist()
                print(tokenizer.decode(token_id),end='',flush=True)

            print()
           


    test_generate()

    # 训练
    min_eval_loss=None
    for epoch in range(epochs):
        model.train()
        epoch_train=[]
        epoch_eval=[]
        epoch_eval_perfect=[]
        epoch_eval_reject=[]

        # run train
        for batch in train_loader:

                    
            perfect_data=batch['perfect'].to(device=device)
            perfect_mask=batch['perfect_mask'].to(device=device)
            reject_data=batch['reject'].to(device=device)
            reject_mask=batch['reject_mask'].to(device=device)

            loss,_,_=run_once(model,ref_model,perfect_data,perfect_mask,reject_data,reject_mask)
            
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_train.append(loss.item())

            # print(loss.item())
            # print(optimizer.state_dict()['param_groups'][0]['lr']) 
        

        # run eval
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                
                perfect_data=batch['perfect'].to(device=device)
                perfect_mask=batch['perfect_mask'].to(device=device)
                reject_data=batch['reject'].to(device=device)
                reject_mask=batch['reject_mask'].to(device=device)

                loss,perfect_loss,reject_loss=run_once(model,ref_model,perfect_data,perfect_mask,reject_data,reject_mask)
                epoch_eval.append(loss.item())
                epoch_eval_perfect.append(perfect_loss.item())
                epoch_eval_reject.append(reject_loss.item())
                

        # print(f'epoch:{epoch},train loss:{eval_model(model,train_loader)}')
        print(f'epoch:{epoch},train loss:{np.mean(epoch_train)}')
        print(f'epoch:{epoch},eval loss:{np.mean(epoch_eval)}')
        print(f'epoch:{epoch},eval loss:{np.mean(epoch_eval_perfect)}')
        print(f'epoch:{epoch},eval loss:{np.mean(epoch_eval_reject)}')
        # 生成文本
        # print(f'{epoch} generate')
        # if epoch>0 and epoch%5==0:
        if True:
            test_generate()

        now_eval_loss=np.mean(epoch_eval)
        if min_eval_loss is None or now_eval_loss<min_eval_loss:
            out_path=os.path.join(out_dir,f'{epoch}_{now_eval_loss}.pth')
            torch.save(model.state_dict(),out_path)
            min_eval_loss=now_eval_loss

    # print(his_lr)
    # import matplotlib.pyplot as plt
    # plt.plot(his_lr)
    # plt.show()

    # 获取eval数据
    # 获取random generate text


    # 保存模型参数，模型与优化器

def main():
    f_path=r'RawTextData/instruction_data_with_preference.json'
    tokenizer,_,_,train_loader,eval_loader=get_data(f_path,batch_size=8)
    for id,i in enumerate(eval_loader):
        print(i['perfect'].shape)
        print(i['perfect_mask'].shape)
        print(i['reject'].shape)
        print(i['reject_mask'].shape)


        print('perfect'+'-'*100)
        print(tokenizer.decode([j for j in i['perfect'][0].tolist() if j>=0]))
        perfect_data=i['perfect']
        perfect_mask=i['perfect_mask']
        mask_data=perfect_data*(1-perfect_mask)+perfect_mask*-1
        print('perfect_masked'+'-'*100)
        print(tokenizer.decode([j for j in mask_data[0].tolist() if j>=0]))


        print('reject'+'-'*100)
        print(tokenizer.decode([j for j in i['reject'][0].tolist() if j>=0]))
        reject_data=i['reject']
        reject_mask=i['reject_mask']
        mask_data=reject_data*(1-reject_mask)+reject_mask*-1
        print('reject_masked'+'-'*100)
        print(tokenizer.decode([j for j in mask_data[0].tolist() if j>=0]))


        # print('-'*100)
        # print(id)
        # input_list,target_list=i

        # for j_id in range(len(input_list)):
        #     input_str=tokenizer.decode(input_list[j_id].tolist())
        #     output_str=tokenizer.decode([j for j in target_list[j_id].tolist() if j>=0])
        #     print('input_str\n',input_str)
        #     print('output_str\n',output_str)

        # print(input_list.shape)
        # print(target_list.shape)
        if id>3:
            break
        print('\n\n')
   
if __name__=='__main__':
    # main()
    train()