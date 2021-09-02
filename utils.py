import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import scipy

from Metrics import cluster_metrics as CM


class WSD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z2, z1):
        l = z2.shape[0]
        embedding_loss = 0
        for i in range(l):
            embedding_loss += scipy.stats.wasserstein_distance(
                z2[i].detach().cpu(), z1[i].detach().cpu())

        embedding_loss = torch.tensor(embedding_loss, requires_grad=True)/l
        return 1-embedding_loss


def pretrain(args, model, pretrain_loader, device, optimizer, criterion_mse, criterion_ssim, criterion_emb, epoch):
    model.train()
    loop = tqdm(enumerate(pretrain_loader),
                total=len(pretrain_loader), leave=False)
    total_recon_loss = 0
    for idx, (x, _, _) in loop:
        x = x.to(device)
        optimizer.zero_grad()
        Gx_bar, Lx_bar, Gz, Lz = model(x)

        Gmse_loss = criterion_mse(Gx_bar, x)
        Gssim_loss = 1 - criterion_ssim(Gx_bar, x)
        Grecon_loss = args.A*Gmse_loss + (1-args.A)*Gssim_loss
        total_recon_loss += Grecon_loss.item()

        Lmse_loss = criterion_mse(Lx_bar, x)
        Lssim_loss = 1 - criterion_ssim(Lx_bar, x)
        Lrecon_loss = args.A*Lmse_loss + (1-args.A)*Lssim_loss
        total_recon_loss += Lrecon_loss.item()

        emb_loss = criterion_emb(Gz, Lz)

        total_loss = Grecon_loss + Lrecon_loss + emb_loss
        # total_loss = emb_loss
        total_loss.backward()

        optimizer.step()

        if idx % args.logint == 0:
            loop.set_description(f"[{epoch}/{args.preepochs}]:")
            loop.set_postfix(GRecon_loss=Grecon_loss.item(), Emb_loss=emb_loss.item(),
                             LRecon_loss=Lrecon_loss.item(), Total_loss=total_loss.item())

            # loop.set_postfix(Emb_loss=emb_loss.item())

            # print(idx)
            # print(model.encoder1.enc[0][0].weight ==model.encoder2.enc[0][0].weight)



def train(args, model, dataset, train_loader, device, optimizer, criterion_mse, criterion_ssim, epoch):
    model.eval()
    X = dataset.x
    y = dataset.y

    with torch.no_grad():

        # X = torch.Tensor(X).to(device).reshape(X.shape[0], -1)
        X = torch.Tensor(X).to(device)
        Gx_bar, Lx_bar, Gz, Lz = model(X)

        Gz_metric = CM.all_metrics(Gz.data.cpu().numpy(), y, 10, 20, -1)
        Lz_metric = CM.all_metrics(Lz.data.cpu().numpy(), y, 10, 20, -1)

        Gz_scores = Gz_metric.scores()
        Lz_scores = Lz_metric.scores()

        print("\n Gz")
        for item, key in Gz_scores.items():
            print(item, key)

        print("\n Lz")
        for item, key in Lz_scores.items():
            print(item, key)


def load_mnist(path="/home/tejas/experimentations/IDEC-pytorch/data/mnist.npz"):
    f = np.load(path)

    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)

    x = np.expand_dims(x, axis=1).astype(np.float32)
    # x = x.reshape((x.shape[0], -1)).astype(np.float32)
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y


def load_fmnist(path="/home/tejas/experimentations/IDEC-pytorch/data/fmnist.npz"):
    f = np.load(path)

    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)

    x = np.expand_dims(x, axis=1).astype(np.float32)
    # x = x.reshape((x.shape[0], -1)).astype(np.float32)
    # x = np.divide(x, 255.)
    print('FMNIST samples', x.shape)
    return x, y


def load_cifar10(path="/home/tejas/experimentations/IDEC-pytorch/data/cifar10.npz"):
    f = np.load(path)

    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)

    # x = np.expand_dims(x,axis=1).astype(np.float32)
    # x = x.reshape((x.shape[0], -1)).astype(np.float32)
    # x = np.divide(x, 255.)
    print('cifar10 samples', x.shape)
    return x, y


class MnistDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_mnist()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


class FMnistDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_fmnist()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


class Cifar10Dataset(Dataset):

    def __init__(self):
        self.x, self.y = load_cifar10()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    # from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
