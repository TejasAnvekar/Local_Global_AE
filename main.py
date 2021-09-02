from utils import MnistDataset, FMnistDataset, Cifar10Dataset
import os
import utils
import config
from network import LG_VAE,LG_VAE2,LG_AE
import torch
import torch.nn as nn
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":

    args = config.config()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = MnistDataset()

    model = LG_AE(
        inp=784,
        L_enc_layers=[50,4],
        G_enc_layers=[500,10],
        dec_layers=[500,784]
    ).to(device)

    pretrain_loader = DataLoader(
        dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=6)
    train_loader = DataLoader(
        dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=6)

    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion_mse = nn.MSELoss().to(device)
    criterion_ssim = SSIM(data_range=1.0, channel=1,
                          size_average=True).to(device)
    # criterion_emb =  nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean').to(device)
    criterion_emb = nn.MSELoss().to(device)

    print(model)

    # checkpoints = torch.load("/home/cvgws-06/Desktop/Tejas/CEVI_SDP-main/Model/fmnist/initial.pth.tar")

    # model.load_state_dict(checkpoints['weights'])

    for epoch in range(args.preepochs):
        utils.pretrain(args, model, pretrain_loader, device,
                                    pretrain_optimizer, criterion_mse, criterion_ssim, criterion_emb, epoch)

        if (epoch + 1) % args.recentre == 0:
            utils.train(args, model, dataset, train_loader, device,
                        train_optimizer, criterion_mse, criterion_ssim, epoch)

        # if (epoch+1)%100 == 0:
        #     k = os.path.join(args.save,args.dataset)
        #     if not  os.path.exists(k):
        #         os.mkdir(k)

        #     torch.save(
        #         {
        #             "weights":model.state_dict(),
        #             "optimizer":pretrain_optimizer.state_dict(),
        #             "epoch":epoch+1,
        #             "Recon_loss":recon_loss,

        #         },os.path.join(k,args.trail+".pth.tar")
        #         )
