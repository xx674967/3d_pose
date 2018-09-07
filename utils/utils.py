from numpy.random import randn
import ref
import torch
from eval import getPreds
import numpy as np





# multi source discriminator



def MSdiscriminator(heatmap,reg,batch):
    hm_2d_numpy = (heatmap.data).cpu().numpy()
    hm_2d = getPreds(hm_2d_numpy);

    reg_3d = (reg.data).cpu().numpy()
    if reg_3d.shape[-1] == 3:
        reg_3d_16x1 = reg_3d[:,:,2:3]
    else:
        reg_3d_16x1 = reg_3d.reshape(batch,16,1)
    reg_3d_16x1 = reg_3d_16x1.astype(np.float64)

    reg_3d_64 = (reg_3d_16x1+1)/2. * 64
    hm_16x3 = np.concatenate([hm_2d,reg_3d_64],axis=2)

    GD_result = GeometricDescriptor(hm_16x3)
    GD_result = torch.from_numpy(GD_result)
    GD_result = torch.autograd.Variable(GD_result).float().cuda()

    hm_depth_result = hm_depthmap(hm_2d,reg_3d_16x1,batch)
    if hm_2d_numpy.shape !=(batch,16,64,64):
        a = hm_2d_numpy.copy()
        d = np.concatenate([a,hm_2d_numpy],axis=0)
        hm_2d_numpy = np.concatenate([d,hm_2d_numpy],axis=0)
        #hm_2d_numpy = np.zeros(batch,16,64,64)
        #print hm_2d_numpy.shape
    hm_2d_3d_result = np.concatenate([hm_2d_numpy,hm_depth_result],axis=1)

    hm_2d_3d_result = torch.from_numpy(hm_2d_3d_result)
    hm_2d_3d_result = torch.autograd.Variable(hm_2d_3d_result).float().cuda()

    return GD_result,hm_2d_3d_result

def GeometricDescriptor(ary):
    result = []
    for bs in ary:
        tem = []
        for i in bs:
            sub_tem = []
            for j in bs:
                sub_tem.append([i[0]-j[0],i[1]-j[1],i[2]-j[2],(i[0]-j[0])**2,(i[1]-j[1])**2,(i[2]-j[2])**2])
            tem.append(sub_tem)
        result.append(tem)
    result = np.array(result)
    result = result.transpose(0,3,2,1)
    return result

def hm_depthmap(pre,reg,batch):

    pre1 = np.zeros((batch,16,64,64))
    for j,zz in enumerate(pre):
        for i,xx in enumerate(zz):

            pre1[j][i][int(xx[0])][int(xx[1])] = reg[j][i]

    return pre1
  

  
def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    lr = LR * (0.1 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
             
def Rnd(x):
  return max(-2 * x, min(2 * x, randn() * x))
  
def Flip(img):
  return img[:, :, ::-1].copy()  
  
def ShuffleLR(x):
  for e in ref.shuffleRef:
    x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
  return x
