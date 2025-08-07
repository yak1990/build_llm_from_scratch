from turtle import forward
import torch
import torch.nn as nn

class Toy_MultiHeadAttention(nn.Module):
    def __init__(self,input_dim,output_dim,num_heads,max_length,drop_out=0.1,qkv_bias=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert output_dim%num_heads==0
        self.num_heads=num_heads
        self.head_dim=output_dim//num_heads
        self.head_norm=self.head_dim**0.5

        self.qkv_layer=nn.Linear(input_dim,3*output_dim,bias=qkv_bias)
        self.dropout_layer=nn.Dropout(drop_out)
        
        self.register_buffer('mask',torch.triu(torch.ones(max_length,max_length),diagonal=1))

    def forward(self,x,mask=None):
        B,L,D=x.shape

        qkv=self.qkv_layer(x)
        qkv=qkv.view(B,L,3,self.num_heads,self.head_dim).permute(2,0,3,1,4) # final shape: (3,B,num_heads,L,head_dim)
        q,k,v=qkv.unbind(0) # each shape: (B,num_heads,L,head_dim)

        atten_score=q@k.transpose(2,3)
        atten_score=atten_score/self.head_norm

        if mask is not None:
            atten_score=atten_score.masked_fill(mask==1,-torch.inf)
        else:
            atten_score=atten_score.masked_fill(self.mask[:L,:L]==1,-torch.inf)
        
        atten_score=torch.softmax(atten_score,dim=-1)
        atten_score=self.dropout_layer(atten_score)

        output=atten_score@v
        output=output.transpose(1,2).contiguous().view(B,L,-1)

        return output


def main():
    input_dim=3
    output_dim=8
    num_heads=4
    max_length=32

    model=Toy_MultiHeadAttention(input_dim,output_dim,num_heads,max_length)

    x=torch.randn(16,10,input_dim)
    y=model(x)
    print(y.shape)


if __name__=='__main__':
    main()