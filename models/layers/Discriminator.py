import torch.nn as nn
import math
import numpy as np
import torch


class Discriminator(nn.Module):

    """Discriminator network for 3d pose."""
    def __init__(self):
        super(Discriminator, self).__init__()


        # 3 type source
        # img source
        layersIn = []
        conv_dim = 32
        layersIn.append(nn.Conv2d(3, conv_dim, kernel_size=3, stride=2, padding=1))
        layersIn.append(nn.ReLU(inplace = True))
        for i in range(3):
            layersIn.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=3, stride=2, padding=1))
            layersIn.append(nn.BatchNorm2d(conv_dim * 2))
            layersIn.append(nn.LeakyReLU(0.2))
            layersIn.append(nn.AvgPool2d(kernel_size=2, stride=2))

            conv_dim = conv_dim * 2
        layersIn.append(nn.Conv2d(conv_dim, 256, kernel_size=2, stride=1))


        # Geometric Descriptor source
        layersGD = []
        layersGD.append(nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1))
        layersGD.append(nn.BatchNorm2d(64))
        layersGD.append(nn.LeakyReLU(0.2))
        layersGD.append(nn.AvgPool2d(kernel_size=2, stride=2))

        layersGD.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        layersGD.append(nn.BatchNorm2d(128))
        layersGD.append(nn.LeakyReLU(0.2))
        layersGD.append(nn.AvgPool2d(kernel_size=2, stride=2))

        layersGD.append(nn.Conv2d(128, 256, kernel_size=1, stride=1))
        # 2d heatmap+depthmap source
        layersHM = []
        hm_channels = 64
        layersHM.append(nn.Conv2d(32, hm_channels, kernel_size=3, stride=2, padding=1))
        for j in range(2):
            layersHM.append(nn.Conv2d(hm_channels, hm_channels * 2, kernel_size=3, stride=2, padding=1))
            layersHM.append(nn.BatchNorm2d(hm_channels * 2))
            layersHM.append(nn.LeakyReLU(0.2))
            layersHM.append(nn.AvgPool2d(kernel_size=2, stride=2))
            hm_channels = hm_channels * 2
        layersHM.append(nn.Conv2d(hm_channels, 256, kernel_size=2, stride=1))

        self.Img = nn.Sequential(*layersIn)
        self.GD = nn.Sequential(*layersGD)
        self.HM = nn.Sequential(*layersHM)

        layersFC = []
        layersFC.append(nn.Linear(256*3, 1000))
        layersFC.append(nn.Linear(1000, 256))
        layersFC.append(nn.Linear(256, 1))
        layersFC.append(nn.Sigmoid())

        self.fc = nn.Sequential(*layersFC)



    def forward(self, img, gd, hmdepth):
        fm = self.Img(img)
        gd = self.GD(gd)
        hm = self.HM(hmdepth)
        fm = fm.view(fm.size(0),fm.size(1))
        gd = gd.view(gd.size(0), gd.size(1))
        hm = hm.view(hm.size(0), hm.size(1))
        #test = fm + gd + hm
        test =torch.cat([fm,gd,hm],1)
        out_src = self.fc(test)

        return out_src

if __name__ == '__main__':
    model = Discriminator().cuda()

    img = np.random.rand(6,3,256,256)
    gd = np.random.rand(6,6,16,16)
    test = np.random.rand(6,6,16,16)

    depthmap = np.random.rand(6,32,64,64)

    img = torch.from_numpy(img)
    img = torch.autograd.Variable(img).float().cuda()

    gd = torch.from_numpy(gd)
    gd = torch.autograd.Variable(gd).float().cuda()


    depthmap = torch.from_numpy(depthmap)
    depthmap = torch.autograd.Variable(depthmap).float().cuda()



    xz = model(img,gd,depthmap)
    print xz
