# copy from https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self,emb_dim,hidden_dim,bias=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1=nn.Linear(emb_dim,hidden_dim,bias=bias)
        self.fc2=nn.Linear(emb_dim,hidden_dim,bias=bias)
        self.fc3=nn.Linear(hidden_dim,emb_dim,bias=bias)

    def forward(self,x):
        x_fc1=self.fc1(x)
        x_fc2=self.fc2(x)
        x=F.silu(x_fc1)*x_fc2
        x=self.fc3(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, emb_dim,eps=1e-6,bias=False,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps=eps
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.shift=nn.Parameter(torch.zeros(emb_dim)) if bias else None
    
    def forward(self,x):
        var=x.pow(2).mean(dim=-1,keepdim=True)
        norm_x=x*torch.rsqrt(var+self.eps)

        x=norm_x*self.scale
        if self.shift is not None:
            x=x+self.shift
        
        return x

def generate_rope_params(head_dim,theta_base=10000,context_length=4096):
    assert head_dim%2==0, 'embedding dim must be even'

    inv_freq=theta_base**(-torch.arange(0,head_dim,2)/head_dim)

    position=torch.arange(context_length)

    angles=position[:,None]*inv_freq[None,:]
    angles=torch.cat([angles,angles],dim=-1)

    cos=torch.cos(angles)
    sin=torch.sin(angles)

    return cos,sin

def apply_rope(x,cos,sin):
    B,H,L,D=x.shape
    assert D%2==0, 'embedding dim must be even'

    x1=x[...,:D//2]
    x2=x[...,D//2:]

    cos=cos[:L][None,None,:,:]
    sin=sin[:L][None,None,:,:]

    rotated=torch.cat([-x2,x1],dim=-1)

    x_rotated=x*cos+rotated*sin
    return x_rotated


class GroupMultiHeadAttention(nn.Module):
    def __init__(self,input_dim,num_heads,kv_group_nums,drop_out=0.1,qk_norm=False,qkv_bias=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert input_dim%num_heads==0
        assert num_heads%kv_group_nums==0


        self.num_heads=num_heads
        self.kv_group_nums=kv_group_nums

        self.group_size=self.num_heads//self.kv_group_nums
        self.head_dim=input_dim//num_heads
        self.head_norm=self.head_dim**0.5

        self.q_layer=nn.Linear(input_dim,input_dim,bias=qkv_bias)
        self.k_layer=nn.Linear(input_dim,kv_group_nums*self.head_dim,bias=qkv_bias)
        self.v_layer=nn.Linear(input_dim,kv_group_nums*self.head_dim,bias=qkv_bias)

        self.qk_norm=qk_norm
        if qk_norm:
            self.q_norm=RMSNorm(self.head_dim)
            self.v_norm=RMSNorm(self.head_dim)

        self.dropout_layer=nn.Dropout(drop_out)

        self.out_proj=nn.Linear(input_dim,input_dim)
        

    def forward(self,x,mask,cos,sin):
        B,L,D=x.shape

        q=self.q_layer(x).view(B,L,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k_layer(x).view(B,L,self.kv_group_nums,self.head_dim).transpose(1,2)
        v=self.v_layer(x).view(B,L,self.kv_group_nums,self.head_dim).transpose(1,2)

        if self.qk_norm:
            q=self.q_norm(q)
            k=self.v_norm(k)
        
        q=apply_rope(q,cos,sin)
        k=apply_rope(k,cos,sin)

        k=torch.repeat_interleave(k,self.group_size,dim=1)
        v=torch.repeat_interleave(v,self.group_size,dim=1)

        atten_score=q@k.transpose(2,3)
        atten_score=atten_score/self.head_norm

        atten_score=atten_score.masked_fill_(mask,-torch.inf)
        
        atten_score=torch.softmax(atten_score,dim=-1)
        atten_score=self.dropout_layer(atten_score)

        output=atten_score@v
        output=output.transpose(1,2).contiguous().view(B,L,-1)

        output=self.out_proj(output)

        return output


class TransformBlock(nn.Module):
    def __init__(self,input_dim,num_head,kv_group_nums,hidden_dim,drop_out,qk_norm, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.norm1=RMSNorm(input_dim)
        self.attn=GroupMultiHeadAttention(input_dim,num_head,kv_group_nums,drop_out,qk_norm)
        self.drop_out=nn.Dropout(drop_out)

        self.norm2=RMSNorm(input_dim)
        self.ffn=FeedForward(input_dim,hidden_dim)
        self.drop_out2=nn.Dropout(drop_out)
    
    def forward(self,x,mask,cos,sin):
        short_cut_x=x

        x=self.norm1(x)
        x=self.attn(x,mask,cos,sin)
        x=self.drop_out(x)
        x=x+short_cut_x

        short_cut_x=x
        x=self.norm2(x)
        x=self.ffn(x)
        x=self.drop_out2(x)
        x=x+short_cut_x

        return x

class Qwen(nn.Module):
    def __init__(self,vocab_size,emb_dim,num_head,kv_group_nums,hidden_dim,num_layers,
                    drop_out=0.1,qk_norm=False, 
                    rope_base=10000,context_length=4096,
                    *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.t_embeding=nn.Embedding(vocab_size,emb_dim)

        self.trf_blocks=nn.ModuleList(
            [
                TransformBlock(emb_dim,num_head,kv_group_nums,hidden_dim,drop_out,qk_norm) for _ in range(num_layers)
            ]
        )

        self.fin_norm=RMSNorm(emb_dim)
        self.fin_layer=nn.Linear(emb_dim,vocab_size,bias=False)

        head_dim=emb_dim//num_head
        cos,sin=generate_rope_params(head_dim,rope_base,context_length)

        self.register_buffer('cos',cos,persistent=False)
        self.register_buffer('sin',sin,persistent=False)
    
    def forward(self,x):
        L=x.shape[1]
        mask=torch.triu(torch.ones(L,L,device=x.device),diagonal=1).bool()

        x=self.t_embeding(x)

        for trf in self.trf_blocks:
            x=trf(x,mask,self.cos,self.sin)

        x=self.fin_norm(x)
        x=self.fin_layer(x)

        return x





def main():
    a=Qwen(1000,64,8,2,32,4)
    b=torch.randint(0,1000,(2,10))
    print(a(b).shape)

if __name__=='__main__':
    main()