import torch
from torch import nn as nn
from einops.layers.torch import Rearrange
import copy

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)


class Mixerblock(nn.Module):
    def __init__(self,dim,num_patchs,token_dim,channel_dim,dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patchs,token_dim,dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(nn.LayerNorm(dim),FeedForward(dim,channel_dim,dropout))


    def forward(self,x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)



        return x



    
class MLPMixer(nn.Module):
    def __init__(self,in_channels,dim,num_classes,patch_size,image_size,depth,token_dim,channel_dim):
        super().__init__()

        assert image_size%patch_size==0,'image dim must be divisible by patch size'
        self.num_patch = (image_size//patch_size)**2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels,dim,patch_size,patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(Mixerblock(dim,self.num_patch,token_dim,channel_dim))


        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(nn.Linear(dim,num_classes))



    def forward(self,x):
        x = self.to_patch_embedding(x)

        for mixer in self.mixer_blocks:
            x = mixer(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x) 






def test():
    inp = torch.randn((1,1,28,28)).cuda()

    model = MLPMixer(in_channels=1,image_size=28,patch_size=7,num_classes=100,dim=512,depth=8,token_dim=256,channel_dim=2048).cuda()

    out = model(inp)

    print(out.shape)


# test()






class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)



class FeedForward2(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.0,momentum=0.99):
        super().__init__()


        self.expo = EMA(momentum)

        self.net1 = nn.Sequential(
            nn.Linear(dim,hidden_dim//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4,dim),
            nn.Dropout(dropout)
        )


        self.net2 = copy.deepcopy(self.net1)

        update_moving_average(self.expo,self.net2,self.net1)


    def forward(self,x):
        x1 = self.net1(x)
        x2 = self.net2(x)
        x = x1 + x2
        return x


class Mixerblock2(nn.Module):
    def __init__(self,dim,num_patchs,token_dim,channel_dim,dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward2(num_patchs,token_dim,dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(nn.LayerNorm(dim),FeedForward2(dim,channel_dim,dropout))


    def forward(self,x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)



        return x



    
class MLPMixer2(nn.Module):
    def __init__(self,in_channels,dim,num_classes,patch_size,image_size,depth,token_dim,channel_dim):
        super().__init__()

        assert image_size%patch_size==0,'image dim must be divisible by patch size'
        self.num_patch = (image_size//patch_size)**2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels,dim,patch_size,patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(Mixerblock2(dim,self.num_patch,token_dim,channel_dim))


        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(nn.Linear(dim,num_classes))



    def forward(self,x):
        x = self.to_patch_embedding(x)

        for mixer in self.mixer_blocks:
            x = mixer(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x) 






def test2():
    inp = torch.randn((1,1,28,28)).cuda()

    model = MLPMixer2(in_channels=1,image_size=28,patch_size=7,num_classes=100,dim=512,depth=8,token_dim=256,channel_dim=2048).cuda()

    out = model(inp)

    print(out.shape)


# test2()