import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm


class encoder(nn.Module):
    def __init__(self, inp, layers=[1000, 500, 100, 10]):
        super().__init__()
        self.enc = nn.ModuleList()

        for layer in layers:
            x = [nn.Linear(inp, layer), nn.LayerNorm(layer), nn.ReLU()]
            if layer == layers[-1]:
                x = [nn.Linear(inp, layer)]

            self.enc.append(nn.Sequential(*x))
            inp = layer

    def forward(self, x):
        for enc in self.enc:
            x = enc(x)
        return x


class decoder(nn.Module):
    def __init__(self, inp, layers=[10, 100, 500, 1000]):
        super().__init__()

        self.dec = nn.ModuleList()

        for layer in layers:
            x = [nn.Linear(inp, layer), LayerNorm(layer), nn.ReLU()]
            if layer == layers[-1]:
                x = [nn.Linear(inp, layer), nn.Sigmoid()]

            self.dec.append(nn.Sequential(*x))
            inp = layer

    def forward(self, x):
        for dec in self.dec:
            x = dec(x)
        return x



def copy_weights(ma_model,current_model):
    for current_params,ma_params in zip(current_model.parameters(),ma_model.parameters()):
        ma_params.data=current_params.data




class LG_AE(nn.Module):
    def __init__(self, inp=784, L_enc_layers=[50, 4], G_enc_layers=[500, 10], dec_layers=[500, 784]):
        super().__init__()
        patch_size = inp//4
        self.Global_enc = encoder(inp=inp, layers=G_enc_layers)

        self.Global_dec = decoder(inp=G_enc_layers[-1], layers=dec_layers)
        self.Local_dec = decoder(inp=G_enc_layers[-1], layers=dec_layers)


        self.Local_enc1 = encoder(inp=patch_size, layers=L_enc_layers)
        self.Local_enc2 = self.Local_enc1
        self.Local_enc3 = self.Local_enc1
        self.Local_enc4 = self.Local_enc1


        self.Local_latent_combiner = nn.Linear(
            in_features=L_enc_layers[-1]*4, out_features=G_enc_layers[-1], bias=False)



    def flatten(self,y):
        return y.reshape(y.shape[0],-1)

    def forward(self,x):
        x1 = self.flatten(x[:,:,0:x.shape[2]//2,0:x.shape[3]//2])
        x2 = self.flatten(x[:,:,x.shape[2]//2:,0:x.shape[3]//2])
        x3 = self.flatten(x[:,:,0:x.shape[2]//2,x.shape[3]//2:])
        x4 = self.flatten(x[:,:,x.shape[2]//2:,x.shape[3]//2:])

        x = self.flatten(x)

        global_z = self.Global_enc(x)
        X_hat = self.Global_dec(global_z)
        X_hat = X_hat.reshape(X_hat.shape[0],1,28,28)

        z1 = self.Local_enc1(x1)
        # copy_weights(self.Local_enc2,self.Local_enc1)
        # copy_weights(self.Local_enc3,self.Local_enc1)
        # copy_weights(self.Local_enc4,self.Local_enc1)
        z2 = self.Local_enc2(x2)
        z3 = self.Local_enc3(x3)
        z4 = self.Local_enc4(x4)

        local_z = torch.cat([z1,z2,z3,z4],dim=1)
        local_z = self.Local_latent_combiner(local_z)

        x_hat = self.Local_dec(local_z)
        x_hat = x_hat.reshape(x_hat.shape[0],1,28,28)

        return X_hat,x_hat,global_z,local_z









class LG_VAE(nn.Module):
    def __init__(self, inp=784, L_enc_layers=[50, 4], G_enc_layers=[500, 10], dec_layers=[500, 784]):
        super().__init__()
        patch_size = inp//4
        self.Global_enc = encoder(inp=inp, layers=G_enc_layers)

        self.Global_dec = decoder(inp=G_enc_layers[-1], layers=dec_layers)
        self.Local_dec = decoder(inp=G_enc_layers[-1], layers=dec_layers)


        self.Local_enc1 = encoder(inp=patch_size, layers=L_enc_layers)
        self.Local_enc2 = self.Local_enc1
        self.Local_enc3 = self.Local_enc1
        self.Local_enc4 = self.Local_enc1

        self.Local_mu1 = nn.Linear(L_enc_layers[-1], L_enc_layers[-1])
        self.Local_mu2 = self.Local_mu1
        self.Local_mu3 = self.Local_mu1
        self.Local_mu4 = self.Local_mu1

        self.Local_var1 = nn.Linear(L_enc_layers[-1], L_enc_layers[-1])
        self.Local_var2 = self.Local_var1
        self.Local_var3 = self.Local_var1
        self.Local_var4 = self.Local_var1

        self.Local_latent_combiner = nn.Linear(
            in_features=L_enc_layers[-1]*4, out_features=G_enc_layers[-1], bias=False)

        self.Global_mu = nn.Linear(G_enc_layers[-1], G_enc_layers[-1])
        self.Global_var = nn.Linear(G_enc_layers[-1], G_enc_layers[-1])


    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return std*eps + mu

    def flatten(self,y):
        return y.reshape(y.shape[0],-1)

    def forward(self,x):
        x1 = self.flatten(x[:,:,0:x.shape[2]//2,0:x.shape[3]//2])
        x2 = self.flatten(x[:,:,x.shape[2]//2:,0:x.shape[3]//2])
        x3 = self.flatten(x[:,:,0:x.shape[2]//2,x.shape[3]//2:])
        x4 = self.flatten(x[:,:,x.shape[2]//2:,x.shape[3]//2:])

        x = self.flatten(x)

        x = self.Global_enc(x)
        global_mu = self.Global_mu(x)
        global_var = self.Global_var(x)

        global_z = self.reparameterize(global_mu,global_var)

        X_hat = self.Global_dec(global_z)
        X_hat = X_hat.reshape(X_hat.shape[0],1,28,28)

        x1 = self.Local_enc1(x1)
        copy_weights(self.Local_enc2,self.Local_enc1)
        copy_weights(self.Local_enc3,self.Local_enc1)
        copy_weights(self.Local_enc4,self.Local_enc1)
        x2 = self.Local_enc2(x2)
        x3 = self.Local_enc3(x3)
        x4 = self.Local_enc4(x4)


        mu1 = self.Local_mu1(x1)
        var1 = self.Local_var1(x1)
        z1 = self.reparameterize(mu1,var1)

        mu2 = self.Local_mu2(x2)
        var2 = self.Local_var2(x2)
        z2 = self.reparameterize(mu2,var2)

        mu3 = self.Local_mu3(x3)
        var3 = self.Local_var3(x3)
        z3 = self.reparameterize(mu3,var3)

        mu4 = self.Local_mu4(x4)
        var4 = self.Local_var4(x4)
        z4 = self.reparameterize(mu4,var4)

        local_z = torch.cat([z1,z2,z3,z4],dim=1)
        local_z = self.Local_latent_combiner(local_z)

        x_hat = self.Local_dec(local_z)
        x_hat = x_hat.reshape(x_hat.shape[0],1,28,28)

        return X_hat,x_hat,global_z,local_z





class LG_VAE2(nn.Module):
    def __init__(self, inp=784, L_enc_layers=[50, 4], G_enc_layers=[500, 10], dec_layers=[500, 784]):
        super().__init__()
        patch_size = inp//4
        self.Global_enc = encoder(inp=inp, layers=G_enc_layers)

        self.Global_dec = decoder(inp=G_enc_layers[-1], layers=dec_layers)
        self.Local_dec = decoder(inp=G_enc_layers[-1], layers=dec_layers)


        self.Local_enc1 = encoder(inp=patch_size, layers=L_enc_layers)
        self.Local_enc2 = self.Local_enc1
        self.Local_enc3 = self.Local_enc1
        self.Local_enc4 = self.Local_enc1

        self.Local_mu = nn.Linear(G_enc_layers[-1], G_enc_layers[-1])
        self.Local_var = nn.Linear(G_enc_layers[-1], G_enc_layers[-1])


        self.Local_latent_combiner = nn.Linear(
            in_features=L_enc_layers[-1]*4, out_features=G_enc_layers[-1], bias=False)

        self.Global_mu = nn.Linear(G_enc_layers[-1], G_enc_layers[-1])
        self.Global_var = nn.Linear(G_enc_layers[-1], G_enc_layers[-1])


    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return std*eps + mu

    def flatten(self,y):
        return y.reshape(y.shape[0],-1)

    def forward(self,x):
        x1 = self.flatten(x[:,:,0:x.shape[2]//2,0:x.shape[3]//2])
        x2 = self.flatten(x[:,:,x.shape[2]//2:,0:x.shape[3]//2])
        x3 = self.flatten(x[:,:,0:x.shape[2]//2,x.shape[3]//2:])
        x4 = self.flatten(x[:,:,x.shape[2]//2:,x.shape[3]//2:])

        x = self.flatten(x)

        x = self.Global_enc(x)
        global_mu = self.Global_mu(x)
        global_var = self.Global_var(x)

        global_z = self.reparameterize(global_mu,global_var)

        X_hat = self.Global_dec(global_z)
        X_hat = X_hat.reshape(X_hat.shape[0],1,28,28)

        x1 = self.Local_enc1(x1)
        copy_weights(self.Local_enc2,self.Local_enc1)
        copy_weights(self.Local_enc3,self.Local_enc1)
        copy_weights(self.Local_enc4,self.Local_enc1)
        x2 = self.Local_enc2(x2)
        x3 = self.Local_enc3(x3)
        x4 = self.Local_enc4(x4)

        local_x = torch.cat([x1,x2,x3,x4],dim=1)
        local_x = self.Local_latent_combiner(local_x)

        local_mu = self.Local_mu(local_x)
        local_var = self.Local_var(local_x)

        local_z = self.reparameterize(local_mu,local_var)

        x_hat = self.Local_dec(local_z)
        x_hat = x_hat.reshape(x_hat.shape[0],1,28,28)

        return X_hat,x_hat,global_z,local_z    

        


if __name__ == "__main__":
    # test_enc()
    # test_dec()
    # test()
    pass
