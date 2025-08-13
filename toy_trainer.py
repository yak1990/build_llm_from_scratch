from torch._dynamo.variables.torch import tracing_state_functions
import text_loader
import toy_llm
import torch
import tiktoken

def eval_model(model,dataloader):
    model.eval()
    loss_list=[]
    with torch.no_grad():
        for batch in dataloader:
            x,y=batch
            x=x.cuda()
            y=y.cuda()
            y_len=len(y.view(-1))
            outputs=model(x)
            loss=torch.nn.functional.cross_entropy(outputs.view(y_len,-1),y.view(-1))
            loss_list.append(loss.item())
    model.train()
    return sum(loss_list)/len(loss_list)

def random_generate(init_txt,tokenizer,model,max_length,top_k=None,temperature=1):
    model.eval()
    model.reset_cache()
    with torch.no_grad():
        
        init_ids=torch.tensor(tokenizer.encode(init_txt))[None,:]
        x=init_ids.cuda()
        for id in range(max_length):
            if id==0:
                logits=model(x,use_cache=True)
            else:
                logits=model(next_id,use_cache=True)
            logits=logits[:,-1]
            logits=logits/temperature
            if top_k is not None:
                top_value,top_id=torch.topk(logits,top_k)
                probs=torch.nn.functional.softmax(top_value,dim=-1)
                next_id=top_id[torch.multinomial(probs,num_samples=1)]
            else:
                next_id=torch.argmax(logits,dim=-1,keepdim=True)
            x=torch.cat([x,next_id],dim=-1)
        
        x=tokenizer.decode(x[0].tolist())
    model.reset_cache()
    model.train()
    return x


def main():
    # 加载数据集，分为训练集、测试集
    def read_file_list(f_list,spilt_eof='<|endoftext|>'):
        raw_txt_list=[]
        for i in f_list:
            with open(i,'r') as f:
                raw_txt_list.append(f.read())
        raw_txt=spilt_eof.join(raw_txt_list)
        return raw_txt
    
    f_list=[
        r'RawTextData/Middlemarch.txt'
    ]
    train_ratio=0.9
    eval_ratio=0.1
    max_length=64
    
    raw_txt=read_file_list(f_list)
    raw_txt_len=len(raw_txt)
    train_txt=raw_txt[:int(raw_txt_len*train_ratio)]
    eval_txt=raw_txt[int(raw_txt_len*train_ratio):]

    # 创建dataloader
    batch_size=32
    stride=max_length//2
    tokenizer=tiktoken.get_encoding("gpt2")
    train_loader=text_loader.create_dataloader(train_txt,batch_size,max_length,stride,tokenizer)
    eval_loader=text_loader.create_dataloader(eval_txt,batch_size,max_length,stride,tokenizer)


    # 设置网络参数
    vocab_size=50297
    d_dim=768
    num_heads=8
    num_layers=8
    drop_out=0.1
    # 初始化网络
    model=toy_llm.Toy_LLM(vocab_size,d_dim,num_heads,max_length,max_length,num_layers,drop_out)
    model.cuda()
    # 设置优化器
    lr=1e-4
    weight_decay=1e-2
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)


    print('init text generate')
    print(random_generate('Hello, I am',tokenizer,model,max_length//2,None))

    # 训练
    epochs=10
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x,y=batch
            x=x.cuda()
            y=y.cuda()
            optimizer.zero_grad()
            outputs=model(x)
            loss=torch.nn.functional.cross_entropy(outputs.view(-1,vocab_size),y.view(-1))
            loss.backward()
            optimizer.step()
        # print(f'epoch:{epoch},train loss:{eval_model(model,train_loader)}')
        print(f'epoch:{epoch},eval loss:{eval_model(model,eval_loader)}')
        # 生成文本
        print(f'{epoch} generate')
        print(random_generate('Hello, I am',tokenizer,model,max_length//2,None))
     

    # 获取eval数据
    # 获取random generate text


    # 保存模型参数，模型与优化器




    pass
   
if __name__=='__main__':
    main()