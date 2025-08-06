from turtle import pos
import torch
from torch.utils.data import Dataset,DataLoader
import tiktoken

class GPTDataset(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride) -> None:
        super().__init__()

        self.input_ids=[]
        self.target_ids=[]

        token_ids=tokenizer.encode(txt,allowed_special={"<|endoftext|>"})

        for i in range(0,len(token_ids)-max_length,stride):
            input_ids=token_ids[i:i+max_length]
            target_ids=token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_ids))
            self.target_ids.append(torch.tensor(target_ids))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index]


def create_dataloader(txt,batch_size,tokenizer,max_length,stride,shuffle=True,drop_last=True,num_workers=0):
    dataset=GPTDataset(txt,tokenizer,max_length,stride)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    return dataloader

def main():
    raw_text="""
    "I like to fancy that Stroud himself would have given it to me, if he'd been able to say what he thought that day."
    And, in answer to a question I put half-mechanically--"Begin again?" he flashed out. "When the one thing that brings me anywhere near him is that I knew enough to leave off?"
    He stood up and laid his hand on my shoulder with a laugh. "Only the irony of it is that I _am_ still painting--since Grindle's doing it for me! The Strouds stand alone, and happen once--but there's no exterminating our kind of art."
    """

    vocab_size=50297
    output_dim=256

    context_length=1024

    token_emb_layer=torch.nn.Embedding(vocab_size,output_dim)
    pos_emb_layer=torch.nn.Embedding(context_length,output_dim)

    batch_size=8
    max_length=16

    tokenizer=tiktoken.get_encoding("gpt2")
    dataloader=create_dataloader(raw_text,batch_size,tokenizer,max_length,max_length//2)

    pos_input=torch.arange(max_length)
    for batch in dataloader:
        x,y=batch

        text_emb=token_emb_layer(x)
        pos_emb=pos_emb_layer(pos_input)

        input_emb=text_emb+pos_emb
        output_emb=token_emb_layer(y)

        print(input_emb.shape)
        print(output_emb.shape)



if __name__=="__main__":
    main()