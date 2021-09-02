import torch 
import torch.nn as nn

class model(nn.Module):
    def __init__(self,cus_num=10000, item_num = 3079, d_embed = 5):
        super(model,self).__init__()
        self.cus_embedding = nn.Embedding(cus_num,d_embed,max_norm=True)
        self.item_embedding = nn.Embedding(item_num,d_embed,max_norm=True)

        self.feed_forward = nn.Sequential(nn.LayerNorm(d_embed),
                                          nn.Linear(d_embed,d_embed*2),
                                          nn.GELU()
                                          nn.Linear(d_embed*2,1),
                                          nn.ReLU())
        self.position_feed_forward = nn.Sequential(nn.LayerNorm(12),
                                                    nn.Linear(12,24),
                                                    nn.GELU(),
                                                    nn.Linear(24,1),
                                                   nn.ReLU())

    def forward(self,cus,item):
        cus = self.cus_embedding(cus)
        item = self.item_embedding(item)

        x = torch.cat((cus,item),dim=1)
        x = torch.transpose(self.feed_forward(x),2,1)
        x = self.position_feed_forward(x)

        return x
