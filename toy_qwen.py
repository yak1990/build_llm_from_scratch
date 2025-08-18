# copy from https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3

from unittest import skip
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from tokenizers import Tokenizer

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

# MOE 占用硬盘、内存、GPU过大，这里不在本地进一步实现了
class MoEFeedForward(nn.Module):
    def __init__(self,num_experts,num_experts_per_tok,emb_dim,hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_experts=num_experts
        self.num_experts_per_tok=num_experts_per_tok

        self.gete=nn.Linear(emb_dim,num_experts,bias=False)
        meta_device=torch.device('meta')
        self.ff_list=nn.ModuleList([
            FeedForward(emb_dim,hidden_dim,bias=False,device=meta_device) for _ in range(num_experts)
        ])
    
    def forward(self,x):
        score=self.gete(x)
        topk_score,topk_indices=torch.topk(score,self.num_experts_per_tok,dim=-1)
        topk_prob=torch.softmax(topk_score)

        expert_output=[]
        for ff in self.ff_list:
            expert_output.append(ff(x).unsqueeze(-2))
        expert_output=torch.cat(expert_output,dim=-2)

        gate_prob=torch.zeros_like(score)
        for i in range(self.num_experts_per_tok):
            id=topk_indices[...,i:i+1]
            prob=topk_prob[...,i:i+1]
            gate_prob.scatter_(dim=-1,index=id,src=prob)
        
        gate_prob=gate_prob.unsqueeze(-1)
        out=(export_output*gate_prob).sum(dim=-2)

        return out



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
    def __init__(self,input_dim,num_heads,kv_group_nums,head_dim=None,drop_out=0.1,qk_norm=False,qkv_bias=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert input_dim%num_heads==0
        assert num_heads%kv_group_nums==0


        self.num_heads=num_heads
        self.kv_group_nums=kv_group_nums

        self.group_size=self.num_heads//self.kv_group_nums
        if head_dim is None:
            self.head_dim=input_dim//num_heads
        else:
            self.head_dim=head_dim
        self.head_norm=self.head_dim**0.5

        self.q_layer=nn.Linear(input_dim,num_heads*self.head_dim,bias=qkv_bias)
        self.k_layer=nn.Linear(input_dim,kv_group_nums*self.head_dim,bias=qkv_bias)
        self.v_layer=nn.Linear(input_dim,kv_group_nums*self.head_dim,bias=qkv_bias)

        self.qk_norm=qk_norm
        if qk_norm:
            self.q_norm=RMSNorm(self.head_dim)
            self.k_norm=RMSNorm(self.head_dim)

        self.dropout_layer=nn.Dropout(drop_out)

        self.out_proj=nn.Linear(num_heads*self.head_dim,input_dim,bias=False)
        

    def forward(self,x,mask,cos,sin,cache_data=None,layer_id=None):
        B,L,D=x.shape

        q=self.q_layer(x).view(B,L,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k_layer(x).view(B,L,self.kv_group_nums,self.head_dim).transpose(1,2)
        v=self.v_layer(x).view(B,L,self.kv_group_nums,self.head_dim).transpose(1,2)

        if self.qk_norm:
            q=self.q_norm(q)
            k=self.k_norm(k)
        
        q=apply_rope(q,cos,sin)
        k=apply_rope(k,cos,sin)

        k=torch.repeat_interleave(k,self.group_size,dim=1)
        v=torch.repeat_interleave(v,self.group_size,dim=1)



        if cache_data is not None:
            cache=cache_data.get(layer_id)
            if cache is not None:
                his_k,his_v=cache
                k=torch.cat([his_k,k],dim=2)
                v=torch.cat([his_v,v],dim=2)
            cache_data.update(layer_id,(k,v))

        # print(layer_id,'q',q[0,0,:3,:5])
        # print(layer_id,'k',k[0,0,:3,:5])
        # print(layer_id,'v',v[0,0,:3,:5])

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
    def __init__(self,input_dim,num_head,kv_group_nums,hidden_dim,drop_out,qk_norm,head_dim=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.norm1=RMSNorm(input_dim)
        self.attn=GroupMultiHeadAttention(input_dim,num_head,kv_group_nums,head_dim=head_dim,drop_out=drop_out,qk_norm=qk_norm)
        self.drop_out=nn.Dropout(drop_out)

        self.norm2=RMSNorm(input_dim)
        self.ffn=FeedForward(input_dim,hidden_dim)
        self.drop_out2=nn.Dropout(drop_out)
    
    def forward(self,x,mask,cos,sin,cache_data=None,layer_id=None):
        short_cut_x=x

        x=self.norm1(x)
        x=self.attn(x,mask,cos,sin,cache_data,layer_id)
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
                    head_dim=None,
                    rope_base=10000,context_length=4096,
                    *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.t_embeding=nn.Embedding(vocab_size,emb_dim)

        self.trf_blocks=nn.ModuleList(
            [
                TransformBlock(emb_dim,num_head,kv_group_nums,hidden_dim,drop_out,qk_norm,head_dim) for _ in range(num_layers)
            ]
        )

        self.fin_norm=RMSNorm(emb_dim)
        self.fin_layer=nn.Linear(emb_dim,vocab_size,bias=False)

        if head_dim is None:
            head_dim=emb_dim//num_head
        cos,sin=generate_rope_params(head_dim,rope_base,context_length)

        self.register_buffer('cos',cos,persistent=False)
        self.register_buffer('sin',sin,persistent=False)
    
    def forward(self,x,cache_data=None):
        L=x.shape[1]

        if cache_data is None:
            mask=torch.triu(torch.ones(L,L,device=x.device),diagonal=1).bool()
            cos,sin=self.cos,self.sin
        else:
            L_cache=cache_data.get_pos()
            if L_cache==0:
                mask=torch.triu(torch.ones(L,L,device=x.device),diagonal=1).bool()
            else:
                col=torch.arange(L_cache+L,device=x.device)
                row=torch.arange(L_cache,L_cache+L,device=x.device)
                mask=row[:,None]<col[None,:]
            cos,sin=self.cos[L_cache:],self.sin[L_cache:]
            cache_data.update_pos(L)

        x=self.t_embeding(x)

        for trf_id,trf in enumerate(self.trf_blocks):
            x=trf(x,mask,cos,sin,cache_data,trf_id)

        x=self.fin_norm(x)
        x=self.fin_layer(x)

        return x


class KV_Cache:
    def __init__(self,layer_num) -> None:
        self.layer_num=layer_num
        self.reset_cache()

    def get(self,layer_id):
        return self.cache[layer_id]
    
    def update(self,layer_id,layer_data):
        self.cache[layer_id]=layer_data

    def get_pos(self):
        return self.pos_id
    
    def update_pos(self,update_length):
        self.pos_id+=update_length

    def reset_cache(self):
        self.cache=[None for _ in range(self.layer_num)]
        self.pos_id=0


class QwenTokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")
    def __init__(self,tokenizer_file_path=None,apply_chat_template=False,add_generation_prompt=False,add_thinking=False) -> None:
        self.apply_chat_template=apply_chat_template
        self.add_generation_prompt=add_generation_prompt
        self.add_thinking=add_thinking

        self._tok=Tokenizer.from_file(tokenizer_file_path)
        self._special_to_id={t:self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token_id=self._special_to_id['<|endoftext|>']
        self.eos_token_id=self.pad_token_id

    def encode(self,text):
        stripped_text=text.strip()

        if stripped_text in self._special_to_id and '\n' not in stripped_text:
            return [self._special_to_id[stripped_text]]
        
        if self.apply_chat_template:
            text=self.warp_chat(stripped_text)
        
        ids=[]
        for parts in filter(None,self._SPLIT_RE.split(text)):
            if parts in self._special_to_id:
                ids.append(self._special_to_id[parts])
            else:
                ids.extend(self._tok.encode(parts).ids)
        return ids
    
    def decode(self,ids):
        return self._tok.decode(ids,skip_special_tokens=False)

    def warp_chat(self,user_msg):
        s=f'<|im_start|>user\n{user_msg}<|im_end|>\n'
        if self.add_generation_prompt:
            s+=f'<|im_start|>assistant'
            if self.add_thinking:
                s+='\n'
            else:
                s+='\n<think>\n\n</think>\n\n'
        return s

def get_qwen():

    from safetensors.torch import load_file
    weight_path=r'Qwen3-0.6B/model.safetensors'
    qwen_weight=load_file(weight_path)

    vocab_size=151936
    emb_dim=1024
    num_head=16
    head_dim=128
    kv_group_num=8
    rope_base=1_000_000
    context_length=40960
    hidden_dim=3072
    num_layer=28
    qk_norm=True

    model=Qwen(vocab_size,emb_dim,num_head,kv_group_num,hidden_dim,num_layer,rope_base=rope_base,context_length=context_length,head_dim=head_dim,qk_norm=qk_norm)

    
    def assign(model_params,weight,weight_name):
        assert model_params.shape==weight[weight_name].shape
        out=nn.Parameter(weight[weight_name])
        return out
    model.t_embeding.weight=assign(model.t_embeding.weight,qwen_weight,'model.embed_tokens.weight')
    model.fin_norm.scale=assign(model.fin_norm.scale,qwen_weight,'model.norm.weight')
    model.fin_layer.weight=assign(model.fin_layer.weight,qwen_weight,'lm_head.weight')
    for id in range(num_layer):
        model.trf_blocks[id].norm1.scale=assign(model.trf_blocks[id].norm1.scale,qwen_weight,f'model.layers.{id}.input_layernorm.weight')
        model.trf_blocks[id].attn.q_layer.weight=assign(model.trf_blocks[id].attn.q_layer.weight,qwen_weight,f'model.layers.{id}.self_attn.q_proj.weight')
        model.trf_blocks[id].attn.q_norm.scale=assign(model.trf_blocks[id].attn.q_norm.scale,qwen_weight,f'model.layers.{id}.self_attn.q_norm.weight')
        model.trf_blocks[id].attn.k_layer.weight=assign(model.trf_blocks[id].attn.k_layer.weight,qwen_weight,f'model.layers.{id}.self_attn.k_proj.weight')
        model.trf_blocks[id].attn.k_norm.scale=assign(model.trf_blocks[id].attn.k_norm.scale,qwen_weight,f'model.layers.{id}.self_attn.k_norm.weight')
        model.trf_blocks[id].attn.v_layer.weight=assign(model.trf_blocks[id].attn.v_layer.weight,qwen_weight,f'model.layers.{id}.self_attn.v_proj.weight')
        model.trf_blocks[id].attn.out_proj.weight=assign(model.trf_blocks[id].attn.out_proj.weight,qwen_weight,f'model.layers.{id}.self_attn.o_proj.weight')
        model.trf_blocks[id].norm2.scale=assign(model.trf_blocks[id].norm2.scale,qwen_weight,f'model.layers.{id}.post_attention_layernorm.weight')
        model.trf_blocks[id].ffn.fc1.weight=assign(model.trf_blocks[id].ffn.fc1.weight,qwen_weight,f'model.layers.{id}.mlp.gate_proj.weight')
        model.trf_blocks[id].ffn.fc2.weight=assign(model.trf_blocks[id].ffn.fc2.weight,qwen_weight,f'model.layers.{id}.mlp.up_proj.weight')
        model.trf_blocks[id].ffn.fc3.weight=assign(model.trf_blocks[id].ffn.fc3.weight,qwen_weight,f'model.layers.{id}.mlp.down_proj.weight')
    

    model=model.float().cuda() 
    
    return model

def generate_text_stream(model,ids,max_new_tokens=100,eos_token_id=None):
    model.eval()
    cache_data=KV_Cache(len(model.trf_blocks))
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out=model(ids,cache_data)[:,-1]
            next_token=out.argmax(dim=-1,keepdim=True)

            if eos_token_id is not None and torch.all(next_token==eos_token_id):
                break
            yield next_token

            if cache_data is None:
                ids=torch.cat([ids,next_token],dim=-1)
            else:
                ids=next_token

def test_generate():
    tokenizer=QwenTokenizer(r'Qwen3-0.6B/tokenizer.json',True,True,True)
    prompt = "Give me a short introduction to large language models."
    prompt='你好，请介绍下自己'
    ids=tokenizer.encode(prompt)
    text=tokenizer.decode(ids)
    print(prompt)
    print(text)

    model=get_qwen()

    ids=torch.tensor(ids).cuda().unsqueeze(0)
    model=model.cuda()
    for token in generate_text_stream(model,ids,max_new_tokens=1000,eos_token_id=tokenizer.eos_token_id):
        token_id=token.squeeze(0).tolist()
        print(tokenizer.decode(token_id),end='',flush=True)


def test_qwen():

    model=get_qwen()
    
    
    
    b=torch.randint(0,1000,(2,10)).cuda()
    print(model(b).shape)

def test_tokenizer():
    tokenizer=QwenTokenizer(r'Qwen3-0.6B/tokenizer.json',True,True,True)
    prompt = "Give me a short introduction to large language models."
    ids=tokenizer.encode(prompt)
    text=tokenizer.decode(ids)
    print(prompt)
    print(text)

def main():
    # test_qwen()
    # test_tokenizer()
    test_generate()


if __name__=='__main__':
    main()