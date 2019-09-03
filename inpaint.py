from __future__ import print_function
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
import torchvision.datasets as dset 
import torchvision.utils as vutils
import torchvision.transforms as transforms
import wgan
import mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train and save GAN or not')
    parser.add_argument('--inpaint', action='store_true', help='Run inpainting')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int,
                        default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64,
                        help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=25,
                        help='number of epochs to train for')
    parser.add_argument('--Diters', type=int, default=5,
                        help='number of D iters per each G iter')
    parser.add_argument('--ziters', type=int, default=5,
                        help='number of z iters per inpainting act')
    parser.add_argument('--lrD', type=float, default=0.00005,
                        help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005,
                        help='learning rate for Generator, default=0.00005')
    parser.add_argument('--nc', type=int, default=3,
                        help='input image channels')
    parser.add_argument('--nz', type=int, default=100,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--n_extra_layers', type=int, default=0,
                        help='Number of extra layers on gen and disc')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--percent', type=float, default=20.0)
    parser.add_argument('--lrz', type=float, default=0.00005,
                        help='learning rate for input, default=0.00005')
    parser.add_argument('--momenz', type=float, default=0.95,
                        help='momentum for SGD optimizer of z')
    parser.add_argument('--lam', type=float, default=0.1,
                        help='weight for perception term in cost function')

    opt = parser.parse_args()

    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and opt.cuda and opt.ngpu > 0) else "cpu")

    # initialiation function for models
    def weight_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # training mode
    if opt.train:
        # dataset select
        dataset = dset.ImageFolder(root=opt.dataroot, 
                                    transform=transforms.Compose([
                                        transforms.CenterCrop((198, 158)), # Crop the edge
                                        transforms.Resize(opt.imageSize),
                                        transforms.CenterCrop(opt.imageSize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                        ]))
        assert dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                                shuffle=True, num_workers=int(opt.workers))
        # wGAN
        netD = wgan.Discriminator(opt.imageSize, opt.nc, opt.ndf, opt.ngpu, opt.n_extra_layers)
        netG = wgan.Generator(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)

        netD.apply(weight_init)
        netG.apply(weight_init)
        
        # device choice
        netD.to(device)
        netG.to(device)
        if (device.type == 'cuda') and (opt.ngpu > 1):
            netD = nn.DataParallel(netD, list(range(opt.ngpu)))
            netG = nn.DataParallel(netG, list(range(opt.ngpu)))

        # optimizer setup
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

        # training epoch
        gen_iter = 0
        dis_iter = 0
        for epoch in range(opt.niter):
            for i, data in enumerate(dataloader, 0):
                if gen_iter<25 or gen_iter%500 == 0:
                    Diter = 100
                else:
                    Diter = opt.Diter
                ################################
                # Train Discriminator network  #
                ################################
                for p in netD.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                # clamp
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
                netD.zero_grad()

                # train with real
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)

                errorD_R = netD(real_cpu)

                # train with fake
                noise = torch.randn((b_size, opt.nz, 1, 1), device=device)
                fake_cpu = netG(noise).detach()
                errorD_F = netD(fake_cpu)

                errorD = errorD_R - errorD_F
                errorD.backward()

                optimizerD.step()
                dis_iter = dis_iter + 1

                ################################
                #   Train Generator network    #
                ################################
                if dis_iter == Diter:
                    for p in netD.parameters():
                        p.requires_grad = False  # to avoid computation
                    
                    netG.zero_grad()
                    noise = torch.randn((opt.batchSize, opt.nz, 1, 1), device=device)
                    fake_cpu = netG(noise)
                    errorG = netD(fake_cpu)
                    errorG.backward()
                    optimizerG.step()

                    dis_iter = 0
                    gen_iter += 1

                    print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                        % (epoch, opt.niter, i, len(dataloader), gen_iter,
                            errorD.data[0], errorG.data[0], errorD_R.data[0], errorD_F.data[0]))

            # do checkpointing
            torch.save(netG.state_dict(),
                    'netG_epoch_{1}.pth'.format(epoch))
            torch.save(netD.state_dict(),
                    'netD_epoch_{1}.pth'.format(epoch))

        # save trained model
        torch.save(netG.state_dict(), 'netG_trained.pth')
        torch.save(netD.state_dict(), 'netD_trained.pth')

    elif opt.inpaint:
        netD = wgan.Discriminator(opt.isize, opt.nc, opt.ndf, opt.ngpu, opt.n_extra_layers)
        netD.load_state_dict(torch.load('netD_trained.pth'))
        netD.eval()
        netG = wgan.Generator(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
        netG.load_state_dict(torch.load('netG_trained.pth'))
        netG.eval()

        transform1 = transforms.Compose([transforms.Grayscale(),
                                        transforms.ToTensor()])
        transform2 = transforms.Compose([transforms.CenterCrop((198, 158)),  # Crop the edge
                                         transforms.Resize(opt.imageSize),
                                         transforms.CenterCrop(opt.imageSize),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])
        imageForMask = dset.ImageFolder(root=opt.dataroot, transform=transform1)
        imageForInpaint = dset.ImageFolder(root=opt.dataroot, transform=transform2)

        dataloaderM = torch.utils.data.DataLoader(imageForMask, batch_size=1,
                                                 shuffle=False, num_workers=opt.workers)
        dataloaderI = torch.utils.data.DataLoader(imageForInpaint, batch_size=1,
                                                  shuffle=False, num_workers=opt.workers)
        M_iter = iter(dataloaderM)
        I_iter = iter(dataloaderI)
        i = 0

        if os.path.isdir('./restore') == False:
            os.mkdir('restore')
        
        while i < len(dataloaderM):
            i += 1
            dataM = M_iter.next()
            dataI = I_iter.next()

            topMask, bottomMask = mask.gen_mask(dataM, opt.percent, 19)
            topW = mask.gen_W(topMask, 7)
            bottomW = mask.gen_W(bottomMask, 7)

            for mask, Wmat, name in zip([topMask,bottomMask], [topW, bottomW], ['top','bottom']):
                z = torch.randn((1, opt.nz, 1, 1), requires_grad=True)
                optimizerZ = optim.SGD(z, lr=opt.lrz, momentum=opt.momenz)
                for k in range(opt.ziters):
                    z.zero_grad()
                    output = netG(z)
                    perception = opt.lam*netD(output)
                    distortion = torch.norm(Wmat*(output-dataI))
                    cost = perception + distortion
                    cost.backward()
                    optimizerZ.step()
                img_gen = netG(z)
                img_restored = mask.post_process(dataI, img_gen, mask)
                img_restored.save('./restore/Img{0}_restored_{1}'.format(i, name))

            ToPIL = transforms.ToPILImage()
            img_ori = ToPIL(dataI.view(3, dataI.size(2), dataI.size(3)))
            img_ori.save('./restore/Img{0}_original'.format(i))


        
