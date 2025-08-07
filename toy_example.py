import text_loader
import toy_llm
import torch

def main():
    batch_size=8
    max_length=16

    dataloader=text_loader.create_example_dataloader(batch_size,max_length,max_length//2)

    vocab_size=50297
    data_dim=256
    
    token_emb_layer=torch.nn.Embedding(vocab_size,data_dim)
    pos_emb_layer=torch.nn.Embedding(max_length,data_dim)
    pos_input=torch.arange(max_length)

    model=toy_llm.Toy_MultiHeadAttention(data_dim,data_dim,8,max_length)

    for batch in dataloader:
        x,y=batch

        text_emb=token_emb_layer(x)
        pos_emb=pos_emb_layer(pos_input)

        input_emb=text_emb+pos_emb
        output=model(input_emb)

        print(input_emb.shape)
        print(output.shape)
        print()

if __name__=='__main__':
    main()