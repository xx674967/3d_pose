import torch
import numpy as np
from utils.utils import AverageMeter,MSdiscriminator,GeometricDescriptor,hm_depthmap
from utils.eval import Accuracy, getPreds, MPJPE
#from utils.debugger import Debugger
from models.layers.FusionCriterion import FusionCriterion
import cv2
import ref
from opts import opts
from progress.bar import Bar
import sys
reload(sys)
sys.setdefaultencoding('UTF-8')

def step(split, epoch, opt, dataLoader,dataLoader_3d,model, criterion, DSmodel, DScriterion, optimizer = None,  DSoptimizer = None):
  if split == 'train':
    model.train()
  else:
    model.eval()
  Loss, Acc, Mpjpe, Loss3D ,D_loss= AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(),AverageMeter()
  
  nIters = len(dataLoader)
  bar = Bar('==>', max=nIters)
  
  
  # for i, (input, target2D, target3D, meta) in enumerate(dataLoader_3d):
  #   input_var_3d = torch.autograd.Variable(input).float().cuda()
  #   target2D_var_3d = torch.autograd.Variable(target2D).float().cuda()
  #   target3D_var_3d = torch.autograd.Variable(target3D).float().cuda()
    

  
  #for i, (input, target2D, target3D, meta) in enumerate(zip(dataLoader,dataLoader_3d)):
  for j,i in zip(enumerate(dataLoader),enumerate(dataLoader_3d)):
  #for i, (input, target2D, target3D, meta),j,(Dinput, Dtarget2D, Dtarget3D, Dmeta) in zip(enumerate(dataLoader), enumerate(dataLoader_3d)):
  #for i, (input, target2D, target3D, meta) in enumerate(dataLoader):

    # input_var = torch.autograd.Variable(input).float().cuda()
    # target2D_var = torch.autograd.Variable(target2D).float().cuda()
    # target3D_var = torch.autograd.Variable(target3D).float().cuda()

    input_var = torch.autograd.Variable(i[1][0]).float().cuda()
    target2D_var = torch.autograd.Variable(i[1][1]).float().cuda()
    target3D_var = torch.autograd.Variable(i[1][2]).float().cuda()

    input_var_or = torch.autograd.Variable(j[1][0]).float().cuda()
    target2D_var_or = torch.autograd.Variable(j[1][1]).float().cuda()
    target3D_var_or = torch.autograd.Variable(j[1][2]).float().cuda()


    # hm_2d_numpy = (target2D_var.data).cpu().numpy()
    # hm_2d = getPreds(hm_2d_numpy)
    # hm_2d_numpy = (target2D_var.data).cpu().numpy()
    # hm_2d = getPreds(hm_2d_numpy)
    #
    # reg_3d = (target3D_var.data).cpu().numpy()
    # reg_3d_16x1 = reg_3d[:,:,2:3]
    # reg_3d_16x1 = reg_3d_16x1.astype(np.float64)
    #
    # reg_3d_64 = (reg_3d_16x1+1)/2. * 64
    # hm_16x3 = np.concatenate([hm_2d,reg_3d_64],axis=2)
    #
    #
    # GD_result = GeometricDescriptor(hm_16x3)
    #
    # GD_result = torch.from_numpy(GD_result)
    # GD_result = torch.autograd.Variable(GD_result).float().cuda()
    #
    #
    #
    #
    # hm_depth_result = hm_depthmap(hm_2d,reg_3d_16x1,opt.trainBatch)
    # hm_2d_3d_result = np.concatenate([hm_2d_numpy,hm_depth_result],axis=1)
    #
    # hm_2d_3d_result = torch.from_numpy(hm_2d_3d_result)
    # hm_2d_3d_result = torch.autograd.Variable(hm_2d_3d_result).float().cuda()
    #
    #
    #


    real_labels = torch.autograd.Variable(torch.ones(opt.trainBatch, 1)).cuda()
    fake_labels = torch.autograd.Variable(torch.zeros(opt.trainBatch, 1)).cuda()

    GD_result,hm_2d_3d_result = MSdiscriminator(target2D_var,target3D_var,opt.trainBatch)

    d_output = model(input_var_or)
    d_reg = d_output[opt.nStack]

    GD_result_or, hm_2d_3d_result_or = MSdiscriminator(d_output[1], d_reg,opt.trainBatch)

    DSoutput_real = DSmodel(input_var,GD_result,hm_2d_3d_result)

    DSoutput_fake = DSmodel(input_var_or, GD_result_or, hm_2d_3d_result_or)


    D_Loss_real = DScriterion(DSoutput_real, real_labels)
    D_Loss_fake = DScriterion(DSoutput_fake, fake_labels)
    D_Loss = D_Loss_fake+D_Loss_real

    D_loss.update(D_Loss.data[0],i[1][0].size(0))
    DSoptimizer.zero_grad()
    D_Loss.backward(retain_variables=True)
    DSoptimizer.step()



    
    output = model(input_var_or)
    reg = output[opt.nStack]
    # if opt.DEBUG >= 2:
    #   print('flag~~~~~~~~~~~~')
    #   gt = getPreds(target2D.cpu().numpy()) * 4
    #   pred = getPreds((output[opt.nStack - 1].data).cpu().numpy()) * 4
    #   debugger = Debugger()
    #   debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
    #   debugger.addPoint2D(pred[0], (255, 0, 0))
    #   debugger.addPoint2D(gt[0], (0, 0, 255))
    #   debugger.showImg()
    #   debugger.saveImg('debug/{}.png'.format(i))

    loss = FusionCriterion(opt.regWeight, opt.varWeight)(reg, target3D_var_or)
    input = j[1][0]
    Loss3D.update(loss.data[0], input.size(0))
    for k in range(opt.nStack):
      loss += criterion(output[k], target2D_var)

    Loss.update(loss.data[0], input.size(0))
    G_Loss_fake = DScriterion(DSoutput_fake, real_labels)
    loss += (1e-4)*G_Loss_fake
    Acc.update(Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (target2D_var.data).cpu().numpy()))
    mpjpe, num3D = MPJPE((output[opt.nStack - 1].data).cpu().numpy(), (reg.data).cpu().numpy(), j[1][3])
    if num3D > 0:
      Mpjpe.update(mpjpe, num3D)
    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    #Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Loss3D {loss3d.avg:.6f} | Acc {Acc.avg:.6f} | Mpjpe {Mpjpe.avg:.6f} ({Mpjpe.val:.6f})'.format(epoch, i[0], nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split=split, Mpjpe=Mpjpe,loss3d=Loss3D)
    Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Loss3D {loss3d.avg:.6f} | D_loss {d_loss.avg:.6f} | Acc {Acc.avg:.6f} | Mpjpe {Mpjpe.avg:.6f} ({Mpjpe.val:.6f})'.format(epoch, i[0], nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split, Mpjpe=Mpjpe, loss3d = Loss3D,d_loss=D_loss)
    bar.next()

  bar.finish()
  return Loss.avg, Acc.avg, Mpjpe.avg, Loss3D.avg
  

def train(epoch, opt, train_loader,train_loader_3d,model, criterion, DSmodel, DScriterion, optimizer, DSoptimizer):
  return step('train', epoch, opt, train_loader, train_loader_3d,model, criterion, DSmodel, DScriterion,optimizer, DSoptimizer)
  
def val(epoch, opt, val_loader, model, criterion):
  return step('val', epoch, opt, val_loader, model, criterion)
