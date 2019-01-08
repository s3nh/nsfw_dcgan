import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
# https://pytorch.org/docs/stable/optim.html
# https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c

from torchvision import transforms
# https://pytorch.org/docs/stable/torchvision/transforms.html
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from pylab import plt
from torchvision import datasets


class Config:
    lr = 0.001
    nz = 64  # redukcja szumu
    image_size = 64
    image_size2 = 64
    nc = 3  # channel of img
    ngf = 64  # generator channel
    ndf = 64  # discriminator channel
    beta1 = 0.5
    batch_size = 32
    max_epoch = 1024
    workers = 2
    gpu = False


# Preprocess
def load_dataset():
    data_path = "./nude_detection/dataset/train_set/"
    dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.RandomResizedCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]))

    dataloader = t.utils.data.DataLoader(dataset, opt.batch_size,
                                           shuffle=True,
                                           num_workers=opt.workers)
    return dataloader


if __name__ == '__main__':
    opt = Config()
    dataloader = load_dataset()
    # Definiowanie modelu

    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#ConvTranspose2d
    netg = nn.Sequential(
        nn.ConvTranspose2d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(opt.ngf * 8),
        nn.ReLU(True),

        nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ngf * 4),
        nn.ReLU(True),

        nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ngf * 2),
        nn.ReLU(True),

        nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ngf),
        nn.ReLU(True),

        nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
        nn.Tanh()

    )

    netd = nn.Sequential(
        nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()

    )

    # Optimizers
    # betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))

    optimizerD = Adam(netd.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = Adam(netg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    criterion = nn.BCELoss()

    fix_noise = Variable(t.FloatTensor(opt.batch_size, opt.nz, 1, 1).normal_(0, 1) )



        # Begin training

    for epoch in range(opt.max_epoch):
        for ii, data in enumerate(dataloader, 0):
            real, _ = data
            input = Variable(real)
            label = Variable(t.ones(input.size(0)))
            noise = t.randn(input.size(0), opt.nz, 1, 1)
            noise = Variable(noise)

            if opt.gpu:
                noise = noise.cuda()
                input = input.cuda()
                label = label.cuda()

            # train discriminator network

            netd.zero_grad()
            output = netd(input)
            error_real = criterion(output.squeeze(), label)
            error_real.backward()

            D_x = output.data.mean()

            # Trenowanie dyskryminatora fake images

            fake_pic = netg(noise).detach()
            output2 = netd(fake_pic)
            label.data.fill_(0)
            error_fake = criterion(output2.squeeze(), label)
            error_fake.backward()
            D_x2 = output2.data.mean()
            error_D = error_real + error_fake
            optimizerD.step()

            # Generator netowrk

            netg.zero_grad()
            label.data.fill_(1)
            noise.data.normal_(0, 1)
            fake_pic = netg(noise)
            output = netd(fake_pic)
            error_G = criterion(output.squeeze(), label)
            error_G.backward()
            optimizerG.step()
            D_G_z2 = output.data.mean()

            if epoch % 100 == 0:
                fake_u = netg(fix_noise)
                imgs = make_grid(fake_u.data * 0.5 + 0.5).cpu()
                im_ = imgs.permute(1, 2, 0).numpy()
                plt.imsave('./output_nsfw/img_{}{}.png'.format(epoch, ii), im_)

                print("Error Generator :  \n {}".format(error_G))
                print("Error Discriminator: \n {}".format(error_D))

    t.save(netd.state_dict(), 'nsfw_dcgan_netd.pth')
    t.save(netg.state_dict(), 'nsfw_dcgan_netg.pth')
