import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

def gen_mask(img, percent, nf):
    with torch.no_grad():
        # Construct the filters
        feature_even = nn.Conv2d(1, nf, 21, 1, 0, bias=False)
        feature_odd = nn.Conv2d(1, nf, 21, 1, 0, bias=False)
        for k in range(nf):
            for i in range(21):
                for j in range(21):
                    theta = 2*np.pi*k/nf
                    x = i*np.cos(theta) + j*np.sin(theta)
                    feature_even.weight.data[k,0,i,j] = np.exp(-(i**2+j**2)/(2*4))*np.cos(np.pi/4*x)
                    feature_odd.weight.data[k,0,i,j] = np.exp(-(i**2+j**2)/(2*4))*np.sin(np.pi/4*x)
        
        # Response for quadrature pairs
        res_even = feature_even(img)
        res_odd  = feature_odd(img)
        res = res_even**2+res_odd**2

        # Rank order and find threshold
        saliency,_ = torch.max(res,1)
        valueOrder, = torch.sort(saliency.view(-1),descending=True)
        pixelNum = valueOrder.size(0)
        targetNum = np.round(pixelNum*percent/100)
        thresholdH = valueOrder[targetNum].item()
        thresholdL = valueOrder[pixelNum-1-targetNum].item()

        # Get mask, size: 1 * (w-dim) * (h-dim)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(64),
                                        transforms.CenterCrop(64),
                                        transforms.ToTensor()])
        top_mask = transform(saliency>thresholdH)
        bottom_mask = transform(saliency<thresholdL)

        top_mask = top_mask.view(1, 1, top_mask.size(1), top_mask.size(2))
        bottom_mask = bottom_mask.view(1, 1, bottom_mask.size(1), bottom_mask.size(2))
        return top_mask.float(), bottom_mask.float()

def gen_W(mask, window):
    with torch.no_grad():
        conv_layer = nn.Conv2d(1, 1, window, 1, (window-1)/2, bias=false)
        conv_layer.weight.data = torch.ones_like(conv_layer.weight.data)

        W = 1 - conv_layer(mask)/window**2
        W = W*mask
    
    Wmat = torch.empty(1, 3, mask.size(2), mask.size(3))
    for i in range(3):
        Wmat[1, i, :, :] = W[1, 1, :, :]
    return Wmat

def post_process(ori, gen, mask):
    ori_img = ori.view((3, ori.size(2), ori.size(3)))
    gen_img = gen.view((3, gen.size(2), gen.size(3)))
    msk_img = torch.empty((3, mask.size(2), mask.size(3)))
    for i in range(3):
        msk_img[i, :, :] = mask[1, 1, :, :]
    res_img = ori_img*msk_img + gen_img*(1-msk.img)
    
    toPIL = transforms.ToPILImage()
    return toPIL(res_img)