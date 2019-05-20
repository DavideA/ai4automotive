import os.path
from argparse import ArgumentParser

import skimage.io as io
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets.dreyeve import DREYEVE
from model.objective import KLD
from model.single_branch import SingleBranchModel
from utils import device
from utils import set_random_seed
from utils import to_image

for d in ['ckps', 'out']:
    if not os.path.exists(d):
        os.makedirs(d)

def parse_args():

    parser = ArgumentParser('DR(eye)VE dataset training')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=20)

    return parser.parse_args()


def train_epoch(ds, model, kld, optim, epoch, args):
    ds.train()
    model.train()
    dl = DataLoader(ds, args.batch_size, num_workers=args.num_workers,
                    worker_init_fn=set_random_seed)
    for batch_idx, data in enumerate(dl):
        x_res, x_crp, x_ff, y, y_crp = (d.to(device) for d in data)

        # DO STUFF HERE - perform a training update


        # print(f'Train epoch: {epoch} '
        #       f'[{batch_idx * len(x_res)}/{len(ds)} '
        #       f'({100. * batch_idx / len(dl):.0f}%)]\t'
        #       f'Loss crop: {loss_crp.item():.4f}\t'
        #       f'Loss resize: {loss_res.item():.4f}\t')


def visualize(ds, model, args, epoch):
    ds.train()
    model.eval()
    dl = DataLoader(ds, args.batch_size, num_workers=args.num_workers,
                    worker_init_fn=set_random_seed)
    with torch.no_grad():
        x_res, x_crp, x_ff, y_res, y_crp = (d.to(device) for d in next(iter(dl)))

        # DO STUFF HERE - get the network prediction

        # image = to_image(p_res, y_res, x_ff)
        # io.imsave(f'out/{epoch:03d}.png', image)


def main():
    args = parse_args()
    ds = DREYEVE()
    model = SingleBranchModel().to(device)
    loss = KLD().to(device)
    optimizer = Adam(model.parameters(), args.lr)

    for epoch in range(0, args.n_epochs):
        visualize(ds, model, args, epoch)
        train_epoch(ds, model, loss, optimizer, epoch, args)
        torch.save(model.state_dict(), f'ckps/{epoch:03d}.pt')


if __name__ == '__main__':
    main()