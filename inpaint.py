from __future__ import print_function
import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
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
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights to initialize')
    parser.add_argument('--pre_D', default='netD_trained.pth',
                        help='Trained weights for netD')
    parser.add_argument('--pre_G', default='netG_trained.pth',
                        help='Trained weights for netG')
    parser.add_argument('--inpaint', action='store_true',
                        help='Run inpainting')
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--saveroot', required=True, help='path to save')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int,
                        default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64,
                        help='the height / width of the input image to network')
    parser.add_argument('--filterSize', type=int, default=21, help='the size of filters')
    parser.add_argument('--niter', type=int, default=25,
                        help='number of epochs to train for')
    parser.add_argument('--Diter', type=int, default=5,
                        help='number of D iters per each G iter')
    parser.add_argument('--GPlambda', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--adam', action='store_true', help='allow adam')
    parser.add_argument('--beta1', type=float, default=0, help='hyperparameter for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='hyperparameter for adam')
    parser.add_argument('--ziters', type=int, default=20,
                        help='number of z iters per inpainting act')
    parser.add_argument('--lrD', type=float, default=0.0001,
                        help='learning rate for Critic, default=0.0001')
    parser.add_argument('--lrG', type=float, default=0.0001,
                        help='learning rate for Generator, default=0.0001')
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
    fsize = opt.filterSize
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
                                        transforms.CenterCrop((218-fsize//2*2, 178-fsize//2*2)), # Crop the edge
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

        if opt.pretrained:
            netD.load_state_dict(torch.load(opt.pre_D))
            netG.load_state_dict(torch.load(opt.pre_G))
        
        # device choice
        netD.to(device)
        netG.to(device)
        if (device.type == 'cuda') and (opt.ngpu > 1):
            netD = nn.DataParallel(netD, list(range(opt.ngpu)))
            netG = nn.DataParallel(netG, list(range(opt.ngpu)))

        # optimizer setup
        if opt.adam:
            optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
            optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
        else:
            optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
            optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
        
        # Change path to saveroot
        if not os.path.isdir(opt.saveroot):
            os.mkdir(opt.saveroot)
        os.chdir(opt.saveroot)

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
                # for p in netD.parameters():
                #     p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
                netD.zero_grad()

                # train with real
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)

                errorD_R = netD(real_cpu).mean(0).view(1)

                # train with fake
                noise = torch.randn((b_size, opt.nz, 1, 1), device=device)
                fake_cpu = netG(noise).detach()
                errorD_F = netD(fake_cpu).mean(0).view(1)

                # compute gradient penalty
                epsl = torch.empty_like(real_cpu)
                for b in range(b_size):
                    v = random.random()
                    for c in range(3):
                        epsl[b, c, :, :] = torch.ones((real.size(2), real.size(3)))*v
                inter1 = real_cpu*epsl + fake_cpu*(1-epsl)
                inter2 = fake_cpu*epsl + real_cpu*(1-epsl)
                GP = (torch.norm((netD(inter1)-netD(inter2))/torch.norm((inter1-inter2), dim=(2,3), keepdim=True).sum(1,keepdim=True), dim=(2,3)) - 1)**2
                GP = GP.mean(0).view(1)

                errorD = - errorD_R + errorD_F + opt.GPlambda*GP
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
                    errorG = netD(fake_cpu).mean(0).view(1)*(-1)
                    errorG.backward()
                    optimizerG.step()

                    dis_iter = 0
                    gen_iter += 1

                    print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                        % (epoch, opt.niter, i, len(dataloader), gen_iter,
                            errorD.data[0], errorG.data[0], errorD_R.data[0], errorD_F.data[0]))

            # do checkpointing
            torch.save(netG.state_dict(),
                    'netG_epoch_{0}.pth'.format(epoch))
            torch.save(netD.state_dict(),
                    'netD_epoch_{0}.pth'.format(epoch))

        # save trained model
        torch.save(netG.state_dict(), 'netG_trained.pth')
        torch.save(netD.state_dict(), 'netD_trained.pth')

    elif opt.inpaint:
        netD = wgan.Discriminator(opt.imageSize, opt.nc, opt.ndf, opt.ngpu, opt.n_extra_layers)
        netD.load_state_dict(torch.load(opt.pre_D))
        netD.eval()
        netG = wgan.Generator(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
        netG.load_state_dict(torch.load(opt.pre_G))
        netG.eval()

        transform1 = transforms.Compose([transforms.Grayscale(),
                                        transforms.Resize(2*opt.imageSize+fsize//2*2),
                                        transforms.CenterCrop(2*opt.imageSize+fsize//2*2),
                                        transforms.ToTensor()])
        transform2 = transforms.Compose([transforms.Resize(opt.imageSize+fsize//2*2),
                                         transforms.CenterCrop(opt.imageSize),
                                         transforms.ToTensor()
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
            dataM,_ = M_iter.next()
            dataI,_ = I_iter.next()

            topMask, bottomMask = mask.gen_mask(dataM, fsize, opt.percent, 19)
            topW = mask.gen_W(topMask, 7)
            bottomW = mask.gen_W(bottomMask, 7)

            for mymask, Wmat, name in zip([topMask,bottomMask], [topW, bottomW], ['top','bottom']):
                z = torch.randn((1, opt.nz, 1, 1), requires_grad=True)
                optimizerZ = optim.SGD([z], lr=opt.lrz, momentum=opt.momenz)
                for k in range(opt.ziters):
                    output = netG(z)
                    perception = -opt.lam*netD(output)
                    distortion = torch.norm(Wmat*(output-dataI))
                    cost = perception + distortion
                    cost.backward()
                    optimizerZ.step()
                    z.grad.zero_()
                # To normalize the generator output
                img_gen = netG(z)
                min_v = torch.min(img_gen).item()
                range_v = torch.max(img_gen).item() - min_v
                img_gen = (img_gen - min_v) / range_v

                img_zero = torch.zeros_like(img_gen) 
                img_restored = mask.post_process(dataI, img_zero, mymask)
                img_restored.save('./restore/Img{0}_restored_{1}.jpg'.format(i, name))

            ToPIL = transforms.ToPILImage()
            img_ori = ToPIL(dataI.view(3, dataI.size(2), dataI.size(3)))
            img_ori.save('./restore/Img{0}_original.jpg'.format(i))
    elif opt.test:
        netD = wgan.Discriminator(
            opt.imageSize, opt.nc, opt.ndf, opt.ngpu, opt.n_extra_layers)
        netD.load_state_dict(torch.load(opt.pre_D))
        netD.eval()
        netG = wgan.Generator(opt.imageSize, opt.nz, opt.nc,
                              opt.ngf, opt.ngpu, opt.n_extra_layers)
        netG.load_state_dict(torch.load(opt.pre_G))
        netG.eval()

        with torch.no_grad():
            noise = torch.randn((64, opt.nz, 1, 1))
            fake = netG(noise).detach().cpu()
            fig = plt.figure()
            plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
            plt.savefig('test_gen2.pdf')



        
