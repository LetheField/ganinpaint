import torch
import torch.nn as nn
import torch.nn.parallel

class Discriminator(nn.Module):
    def __init__(self, isize, nc, ndf, ngpu, n_extra_layers=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # Initial input layers
        main = nn.Sequential()
        main.add_module('initial:{0}-{1}:conv'.format(nc,ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:leaky'.format(ndf),
                        nn.LeakyReLU(0.2,inplace=True))
        csize, cndf = isize/2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:bn'.format(t, ndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:leaky'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        # pyramid layers
        while csize > 4:
            in_feat = cndf
            out_feat = cndf*2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:bn'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:leaky'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            
            csize = csize / 2
            cndf = cndf * 2

        # output layer: b_size * cndf * 4 * 4 -> b_size * 1 * 1 * 1
        main.add_module('output:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        # average against mini-batch
        output = output.mean(0)
        return output.view(1)

class Generator(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # find cngf for the first convtranspose layer
        cngf, tisize = ngf//2, 4
        while tisize < isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # initial layers
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:bn'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:ReLU'.format(cngf),
                        nn.ReLU(True))

        cngf, csize = cngf, 4
        # pyramid layers. After these layers, csize will be isize//2
        while csize < isize//2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:bn'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid:{0}:ReLU'.format(cngf//2),
                            nn.ReLU(True))
            
            csize = csize * 2
            cngf = cngf // 2
        
        # extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:bn'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}:{1}:ReLU'.format(t, cngf),
                            nn.ReLU(True))
        # output layers
        main.add_module('output:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('output:{0}:tanh'.format(nc),
                        nn.Tanh())

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output
            