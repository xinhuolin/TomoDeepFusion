from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from model import PConvUNet, NestedUNet, NLD, VGG16FeatureExtractor
from torchvision import transforms
import scipy.io as scio
import os
from PIL import Image
from Recon import Recon
from skimage.transform import radon, iradon, iradon_sart

# use_gpu = torch.cuda.is_available()
# mdoel_G_dir = r"../model_weights/Denoise(49).pth"
# # mdoel_G_dir = r"../model/cycleGAN3/cycle_29B.pth"
# # mdoel_G_dir = r"../model/cycleGAN3/tomo_sino/cycle_49B.pth"
# # mdoel_G_dir = r"../model/denoise/model_weight_2/Denoise(80).pth"
# # use_gpu = False
# if use_gpu:
#     # netGB = NestedUNet().cuda()
#     G_net = NestedUNet().cuda()
#     if torch.cuda.device_count() > 1:
#         netG = nn.DataParallel(G_net)
# else:
#     netG = NestedUNet()
# # netGB.eval()
# netG.load_state_dict(torch.load(mdoel_G_dir))
# transform = transforms.Compose(
#     [transforms.ToTensor()])
#
# path = r"dataadf.mat"
# data = scio.loadmat(path)
# keys = list(data.keys())
# print(keys)
# angles = data[keys[-4]]
# print(angles)
# size = data[keys[-3]].shape
# tomos = {}
# algos = ["wbp", "sart", "sirt"]
# flag = 2
# i = 1
# tomos[keys[i-3]] = np.zeros((size[0], size[1], size[1]))
# for j in tqdm(range(size[0])):
#     sinogram = np.array(data[keys[i-3]][:,j,:])
#     angle = list(angles[0])
#     if flag==0:
#         tomogram = Recon.wbp(sinogram, angles=angle)  # iradon 0. 255.
#     elif flag==1:
#         tomogram = Recon.sart_cuda(sinogram, angles=angle)  # iradon 0. 255.
#     else:
#         tomogram = Recon.sirt_xin(sinogram, angles=angle)  # iradon 0. 255.
#     # tomogram = Recon.sirt_cuda(sinogram, angles=angle)  # iradon 0. 255.
#     # tomogram = Recon.irdm_80(sinogram, angles=angle)  # iradon 0. 255.
#     tomos[keys[i - 3]][j,:,:] = tomogram.copy()
# img = transform(tomos[keys[i - 3]].transpose((2,1,0)))
# ori_tensor = torch.unsqueeze(img.type(torch.cuda.FloatTensor).cuda(), 1)
# with torch.no_grad():
#     rec = netG(ori_tensor)
#     rec = rec.cpu().numpy()[:, 0, :, :]
# scio.savemat(keys[i-3] + "[%s].mat"%algos[flag], {keys[i-3]:tomos[keys[i - 3]]})
# scio.savemat(keys[i-3] + "[%s_denoise].mat"%algos[flag], {keys[i-3]:tomos[keys[i - 3]]})
# scio.savemat(keys[-4] + ".mat", {keys[-4]:angles})



# algos = ["wbp", "sart", "sirt"]
# flag = 0
# path = r"Denoise(49)[%s].mat" % algos[flag]
# denoise_path = r"Denoise(49)_[%s]_denoise.mat" % algos[flag]
# data = scio.loadmat(path)
# data_denoise = scio.loadmat(denoise_path)
# keys = list(data_denoise.keys())
# print(keys)
# tomos = data[keys[-2]]
# print(tomos.shape)
# tomos_den = data_denoise[keys[-2]]
# for x in range(tomos.shape[0]):
#     matplotlib.image.imsave("display/tomodenoise/%s.png"%(x),
#                             tomos_den[x,:,:],
#                             cmap=plt.cm.Greys_r)
#     matplotlib.image.imsave("display/tomo/%s.png" % (x),
#                             tomos[x,:,:],
#                             cmap=plt.cm.Greys_r)

# import struct
# import os
# if __name__ == '__main__':
#     output = np.zeros((128,128,128))
#     filepath='output_datapd.bin'
#     binfile = open(filepath, 'rb') #打开二进制文件
#     size = os.path.getsize(filepath) #获得文件大小
#     for z in range(128):
#         for y in range(128):
#             for x in range(128):
#                 s = binfile.read(4) #每次输出一个字节
#                 output[x,y,z] = struct.unpack('f', s)[0]
#     binfile.close()
#
# print(tomos.shape, output.shape)

path = r"dataadf.mat"
data = scio.loadmat(path)
keys = list(data.keys())
print(keys)
angles = data[keys[-1]]
print(angles)
angle = angles[0].astype(np.float)
for j in tqdm(range(data["adf"].shape[0])):
    sinogram = np.array(data["adf"][j, :, :]).astype(np.float)
    tomogram = iradon(sinogram, theta=angle)
    # tomogram = iradon_sart(sinogram, theta=angle, image=None, relaxation=0.3)

