"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
import model
import matplotlib.pyplot as plt


def train(flow, trainloader, optimizer, epoch):
    flow.train()  # set to training mode
    losses = 0.0
    mses = 0.0
    likelihoods = 0.0
    batches = 0
    for i, (inputs, _) in enumerate(trainloader):
        inputs = inputs.view(inputs.shape[0], inputs.shape[1]*inputs.shape[2]*inputs.shape[3]) #change  shape from BxCxHxW to Bx(C*H*W)
        likelihood, mse = flow(inputs)
        likelihood_mean = -torch.mean(likelihood)
        mse_mean = -torch.mean(mse)
        loss = likelihood_mean - mse_mean
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(flow.parameters(), 1)
        # for p in (p for p in flow.parameters() if p.grad is not None):
        #     torch.nn.utils.clip_grad_norm_(p, 100000000000)
        # print(nll)
        # print([p.grad.data.norm(2).item() for p in flow.parameters() if p.grad is not None])
        optimizer.step()
        flow.zero_grad()
        batches += 1
        losses -= loss.item()
        mses -= mse_mean.item()
        likelihoods -= likelihood_mean.item()
    epoch_mean_loss = losses / batches
    epoch_mean_mse = mses / batches
    epoch_mean_likelihood = likelihoods / batches
    print(f'    Train Mean Loss:{epoch_mean_loss}')
    print(f'    Train Mean MSE:{epoch_mean_mse}')
    print(f'    Train Mean Likelihood:{epoch_mean_likelihood}')
    return epoch_mean_loss




def test(flow, testloader, filename, epoch, sample_shape):
    flow.eval()  # set to inference mode
    with torch.no_grad():
        samples = flow.sample(64).cpu()
        samples = samples.view(-1,sample_shape[0],sample_shape[1],sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
        losses = 0.0
        mses = 0.0
        likelihoods = 0.0
        batches = 0
        for inputs, _ in testloader:
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])
            likelihood, mse = flow(inputs)
            likelihood_mean = -torch.mean(likelihood)
            mse_mean = -torch.mean(mse)
            loss = likelihood_mean - mse_mean
            batches += 1
            losses -= loss.item()
            mses -= mse_mean.item()
            likelihoods -= likelihood_mean.item()
        epoch_mean_loss = losses / batches
        epoch_mean_mse = mses / batches
        epoch_mean_likelihood = likelihoods / batches
        print(f'    Test Mean Loss:{epoch_mean_loss}')
        print(f'    Test Mean MSE:{epoch_mean_mse}')
        print(f'    Test Mean Likelihood:{epoch_mean_likelihood}')
        return epoch_mean_loss


def dequantize(x):
    return  x + torch.zeros_like(x).uniform_(0., 1./256.)

def main(args):
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1, 28, 28]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(dequantize) #dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'coupling%s_' % args.coupling \
             + 'mid%d_' % args.mid_dim \
             + 'hidden%d_' % args.hidden \
             + 'compress%d_' % args.compression_factor \
             + '.pt'

    flow = model.NICE(
                prior=args.prior,
                coupling=args.coupling,
                in_out_dim=784,
                mid_dim=args.mid_dim,
                hidden=args.hidden,
                compress=args.compression_factor,
                device=device).to(device)
    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    train_ll, test_ll = [],[]
    for epoch in range(args.epochs):
        print(f'--Epoch {epoch}--')
        train_ll.append(train(flow, trainloader, optimizer, epoch))
        test_ll.append(test(flow, testloader, model_save_filename, epoch, sample_shape))
    print(train_ll)
    plt.plot(train_ll, label='Train Log Likelihood')
    plt.legend()
    plt.plot(test_ll, label='Test Log Likelihood')
    plt.legend()
    plt.savefig(f'log_likelihood_plot_{args.coupling}_{args.dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    parser.add_argument('--compression_factor',
                        help='the factor of compression by virtual bottleneck.',
                        type=int,
                        default=2)

    args = parser.parse_args()
    main(args)
