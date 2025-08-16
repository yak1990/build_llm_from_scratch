import torch
import torch.nn as nn
import math


class Toy_MultiHeadAttention(nn.Module):
    def __init__(self,input_dim,output_dim,num_heads,windows_size,drop_out=0.1,qkv_bias=False,causal_atten=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert output_dim%num_heads==0
        self.num_heads=num_heads
        self.head_dim=output_dim//num_heads
        self.head_norm=self.head_dim**0.5

        self.qkv_layer=nn.Linear(input_dim,3*output_dim,bias=qkv_bias)
        self.dropout_layer=nn.Dropout(drop_out)

        self.out_proj=nn.Linear(output_dim,output_dim)
        
        if causal_atten:
            self.register_buffer('mask',torch.triu(torch.ones(windows_size,windows_size).bool(),diagonal=1))
        else:
            self.register_buffer('mask',torch.zeros(windows_size,windows_size).bool())

        self.key_cache=None
        self.value_cache=None
        self.cur_id=0

        self.windows_size=windows_size

    def forward(self,x,use_cache=False):
        B,L,D=x.shape

        qkv=self.qkv_layer(x)
        qkv=qkv.view(B,L,3,self.num_heads,self.head_dim).permute(2,0,3,1,4) # final shape: (3,B,num_heads,L,head_dim)
        q,k,v=qkv.unbind(0) # each shape: (B,num_heads,L,head_dim)

        if use_cache:
            if self.key_cache is None or self.key_cache.shape[0]!=B:
                self.key_cache=torch.zeros(B,self.num_heads,self.windows_size,self.head_dim,device=x.device)
                self.value_cache=torch.zeros(B,self.num_heads,self.windows_size,self.head_dim,device=x.device)
                self.cur_id=0
            
            if self.cur_id+L>self.windows_size:
                move_step=self.cur_id+L-self.windows_size
                # torch.roll_(self.key_cache,dims=2,steps=-move_step)
                # torch.roll_(self.value_cache,dims=2,steps=-move_step)

                self.key_cache[:,:,:self.windows_size-move_step,...]=self.key_cache[:,:,move_step:,...]
                self.value_cache[:,:,:self.windows_size-move_step,...]=self.value_cache[:,:,move_step:,...]
                self.cur_id=self.cur_id-move_step
            
            self.key_cache[:,:,self.cur_id:self.cur_id+L,:]=k
            self.value_cache[:,:,self.cur_id:self.cur_id+L,:]=v
            self.cur_id+=L
            
            col_id=torch.arange(self.cur_id,device=x.device)
            row_id=torch.arange(self.cur_id-L,self.cur_id,device=x.device)
            atten_mask=col_id[None,:]>row_id[:,None]


            k=self.key_cache[:,:,:self.cur_id]
            v=self.value_cache[:,:,:self.cur_id]
        else:
            atten_mask=self.mask[:L,:L]
        
        # print('k front',k[0,0,:3])
        # print('k end',k[0,0,-3:])
        # print('v front',v[0,0,:3])
        # print('v end',v[0,0,-3:])
        # print('k',k.shape)
        # print('v',v.shape)


        atten_score=q@k.transpose(2,3)
        atten_score=atten_score/self.head_norm

        atten_score=atten_score.masked_fill_(atten_mask,-torch.inf)
        
        atten_score=torch.softmax(atten_score,dim=-1)
        atten_score=self.dropout_layer(atten_score)

        output=atten_score@v
        output=output.transpose(1,2).contiguous().view(B,L,-1)

        output=self.out_proj(output)

        return output


    def reset_cache(self):
        self.key_cache=None
        self.value_cache=None
        self.cur_id=0

class Toy_LayerNorm(nn.Module):
    def __init__(self,d_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean=nn.Parameter(torch.zeros(d_dim))
        self.std=nn.Parameter(torch.ones(d_dim))
        self.eps=1e-5
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        norm_x=(x-mean)/(std+self.eps)
        out_x=norm_x*self.std+self.mean
        return out_x

class Toy_Gelu(nn.Module):
    def forward(self,x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
        ))

class Toy_FFN(nn.Module):
    def __init__(self,d_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_dim=4*d_dim
        self.net=nn.Sequential(
            nn.Linear(d_dim,hidden_dim),
            Toy_Gelu(),
            nn.Linear(hidden_dim,d_dim)
        )
    
    def forward(self,x):
        return self.net(x)

class Toy_Transformer(nn.Module):
    def __init__(self,d_dim,num_heads,windows_size,drop_out=0.1,qkv_bias=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.attn=Toy_MultiHeadAttention(d_dim,d_dim,num_heads,windows_size,drop_out,qkv_bias)
        self.ffn=Toy_FFN(d_dim)
        self.norm1=Toy_LayerNorm(d_dim)
        self.norm2=Toy_LayerNorm(d_dim)

        self.shortcut_drop_out=nn.Dropout(drop_out)
    
    def forward(self,x,use_cache=False):
        short_cut=x
        x=self.norm1(x)
        x=self.attn(x,use_cache)
        x=self.shortcut_drop_out(x)
        x=x+short_cut

        short_cut=x
        x=self.norm2(x)
        x=self.ffn(x)
        x=self.shortcut_drop_out(x)
        x=x+short_cut
        return x

    def reset_cache(self):
        self.attn.reset_cache()

class Toy_LLM(nn.Module):
    def __init__(self,vocab_size,d_dim,num_heads,windows_size,max_length,num_layers,drop_out=0.1,qkv_bias=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.text_embedding=nn.Embedding(vocab_size,d_dim)
        self.pos_embedding=nn.Embedding(max_length,d_dim)
        self.embedding_dropout=nn.Dropout(drop_out)

        self.transformers=nn.ModuleList([
            Toy_Transformer(d_dim,num_heads,windows_size,drop_out,qkv_bias)
            for _ in range(num_layers)
        ])

        self.fin_norm=Toy_LayerNorm(d_dim)
        self.fin_layer=nn.Linear(d_dim,vocab_size,bias=False)

        self.cur_id=0
        self.windows_size=windows_size

    
    def forward(self,x,use_cache=False):
        B,L=x.shape

        if use_cache:
            pos=torch.arange(self.cur_id,self.cur_id+L,device=x.device)
            self.cur_id+=L
        else:
            pos=torch.arange(L,device=x.device)

        if L>self.windows_size:
            x=x[:,-self.windows_size:]
            pos=pos[-self.windows_size:]

        # print('x',x)
        # print('pos',pos)


        text_emb=self.text_embedding(x)
        pos_emb=self.pos_embedding(pos)

        x=text_emb+pos_emb
        # print('x',x[0,-1,-10:])
        x=self.embedding_dropout(x)

        for transformer in self.transformers:
            x=transformer(x,use_cache)  

        x=self.fin_norm(x)
        logits=self.fin_layer(x)
        return logits

    def reset_cache(self):
        self.cur_id=0
        for i in self.transformers:
            i.reset_cache()

def test_mha():
    input_dim=3
    output_dim=8
    num_heads=4
    max_length=32

    model=Toy_MultiHeadAttention(input_dim,output_dim,num_heads,max_length)

    x=torch.randn(16,10,input_dim)
    y=model(x)
    print(y.shape)

def test_llm():

    torch.manual_seed(10)

    vocab_size=98
    input_dim=16
    num_heads=4
    max_length=128
    windows_size=32
    layer_num=4

    model=Toy_LLM(vocab_size,input_dim,num_heads,windows_size,max_length,layer_num,).cuda()

    x=torch.randint(0,vocab_size,(1,5)).cuda()
    
    y=model(x)
    print(x.shape,y.shape)
    print(x)
    print(y)

    text_len=max_length-10
    model.eval()
    with torch.no_grad():
        import copy
        cache_x=copy.deepcopy(x)


        no_cache={}
        from time import time
        t=time()
        for id in range(text_len):
            print(id,x.shape)
            logits=model(x)
            logits=logits[:,-1]
            # print(id,logits[-1,:5],logits[-1,-5:])
            next_id=torch.argmax(logits,dim=-1,keepdim=True)
            x=torch.cat([x,next_id],dim=-1)
            # print(id,x)
            no_cache[id]=next_id
        print('no cache time:',time()-t)
        
        x=cache_x
        t=time()
        for id in range(text_len):
            print(id,x.shape)
            if id==0:
                logits=model(x,use_cache=True)
            else:
                logits=model(next_id,use_cache=True)
            logits=logits[:,-1]
            # print(id,logits[-1,:5],logits[-1,-5:])
            next_id=torch.argmax(logits,dim=-1,keepdim=True)
            x=torch.cat([x,next_id],dim=-1)
            # print(id,x)
            diff=abs(no_cache[id]-next_id)
            print(id,'diff',max(diff),min(diff))
        print('with cache time:',time()-t)
            






def main():
    test_llm()
   
if __name__=='__main__':
    main()