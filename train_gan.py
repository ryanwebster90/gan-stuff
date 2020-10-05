from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time
import numpy as np
import yaml

def load_config(path):
    # with open(DEFAULT_CONFIG, 'r') as f:
    #     config = yaml.load(f)
    with open(path, 'r') as f:
        config = yaml.load(f)
    # config.update(config_new)
    return config


parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='config')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--ganType', default='lsgan',help='dcgan or wgan')
parser.add_argument('--blockType', default='convnet',help='convnet resnet')
parser.add_argument('--ngpu', type=int,default=1,help='no. gpus to use')
parser.add_argument('--nz', type=int,default=512,help='no. latent dims')
parser.add_argument('--ganArch', default='resnet',help='resnet, simple, prog')
parser.add_argument('--dName', default='celeba',help='name of dataset')
parser.add_argument('--ganReg', default='real',help='fake, real, wgangp or none')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--useNorm', default=True)
parser.add_argument('--N_D_iter',type=int, default=1, help='no. discrim updates')
parser.add_argument('--gpAlpha',type=float, default=1, help='factor for gp loss term')




opt = parser.parse_args()
config = load_config(opt.config)
opt.dataroot = config['root']
opt.imageSize = config['size']
opt.batchSize = config['batchsize']
opt.lr = config['lr']


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

useNorm = opt.useNorm

cudnn.benchmark = True

dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

def update_average(model_tgt, model_src, beta):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
    
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
    
    
device = torch.device("cuda:0")
N_scales = int(np.log2(opt.imageSize))-2

#import resnet
import my_resnet
import copy

netG = my_resnet.Generator(opt.nz,opt.imageSize,block_type=opt.blockType).to(device)
netD = my_resnet.Discriminator(opt.nz,opt.imageSize,block_type=opt.blockType).to(device)
 
netG_test = copy.deepcopy(netG)

z = torch.randn(8,opt.nz,device=device)
y = torch.zeros(8,dtype = torch.long,device=device)
#y = torch.tensor([0],dtype=torch.long,device=device)


criterion = nn.BCELoss()

nz = opt.nz

fn_size = min(32,opt.batchSize)
fixed_noise = torch.randn(fn_size,nz).cuda()
real_label = 1
fake_label = 0

beta1 = .5

# setup optimizer
optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=opt.lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(beta1, 0.999))


N_D_iter = opt.N_D_iter

#restart_id = np.random.randint(0,999)
restart_id = 0

output_folder = './results/'
if os.path.isdir(output_folder) == False:
    os.mkdir(output_folder)

output_folder  = output_folder+f'{opt.config[:-5]}/'

if os.path.isdir(output_folder) == False:
    os.mkdir(output_folder)

bench_file = output_folder + 'a_bench_epochs.txt'


import my_train
trainer = my_train.Trainer(
   netG, netD, optimizerG, optimizerD,
    gan_type=opt.ganType,
    reg_type=opt.ganReg,
    reg_param=opt.gpAlpha,
    use_diffaug=config['diffaug']
    
)


#input('blah2')
nlabels=1

for epoch in range(opt.niter):
    bench_times = open(bench_file,'a')
    t0 = time.perf_counter()
    for i, data in enumerate(dataloader, 0):

        x_real = data[0].to(device)
        batch_size = x_real.size(0)
        
        y = torch.zeros(batch_size,dtype=torch.long,device=device)
        # train with fake
        z = torch.randn(batch_size,opt.nz).cuda()
        
        # add option for D_iter here updates here
        for diter in range(opt.N_D_iter):
            dloss, reg = trainer.discriminator_trainstep(x_real, z)
        
        gloss = trainer.generator_trainstep(z)
        
        update_average(netG_test, netG, beta=.999)
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f GP(D)%.4f\r'
              % (epoch, opt.niter, i, len(dataloader),
                 dloss, gloss,reg), end="")
        if i % 100 == 0:

            y = torch.zeros(fn_size,dtype = torch.long,device=device)
            #y = torch.tensor([0],dtype=torch.long,device=device)
            fake = netG_test(fixed_noise)
            vutils.save_image(fake.detach(),
                    output_folder + 'fake_samples_epoch_%03d.jpg' % (epoch),
                    normalize=True,nrow=4)
        if epoch % 10 ==0 and i==0:
            torch.save(netG_test,output_folder + f'netG_epoch{epoch:02d}.pth')
            
        
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    t_e = t1-t0
    bench_times.write(f'epoch={epoch:04d}, time= {t_e:.04e}\n')
    bench_times.close()
    # do checkpointing
    torch.save(netG_test,output_folder + 'netG_final.pth')
