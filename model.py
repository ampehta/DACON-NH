class model(nn.Module):
    def __init__(self,cus_num=10000, item_num = 3079, d_embed = 5):
        super(model,self).__init__()
        self.cus_embedding = nn.Embedding(cus_num,d_embed,max_norm=True)
        self.item_embedding = nn.Embedding(item_num,d_embed,max_norm=True)

        self.feed_forward = nn.Sequential(nn.LayerNorm(5),
                                          nn.Linear(5,1),
                                          nn.GELU())
        self.position_feed_forward = nn.Sequential(nn.LayerNorm(12),
                                                    nn.Linear(12,1),
                                                    nn.GELU())

    def forward(self,cus,item):
        cus = self.cus_embedding(cus)
        item = self.item_embedding(item)

        x = torch.cat((cus,item),dim=1)
        x = torch.transpose(self.feed_forward(x),2,1)
        x = self.position_feed_forward(x)

        return x
