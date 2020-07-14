import os
import random
import torch
from multiprocessing import Pool
import numpy as np
from PIL import Image
import skimage
from skimage.restoration import denoise_tv_bregman
from skimage.transform import radon, iradon, iradon_sart
from torchvision.transforms import ToTensor
from tqdm import tqdm
import tomopy
from model_structure.model import NestedUNet
import model_structure.unet_standard_3d as unet_standard_3d
from torchvision import transforms
import time

import model_structure.rof as rof

# model_dir = "/data/HeWei/model/denoise/model_weight_2"
dirname = "test_D3"
original_imgs_dir = "../../data/dataset/cycleGAN/%s/original_256" % dirname

def Normalize(np_array):
    return (np_array - np_array.min())/(np_array.max()-np_array.min())

class Recon(object):
    @staticmethod
    def wbp(sinogram_np, angles):
        # print("--start irdm---")
        recon_wb = iradon(sinogram_np, theta=angles, circle=True) / 255.
        return recon_wb

    @staticmethod
    def irdm(sinogram_np, angles, model_dir, numIter=1):
        # print("--start irdm---")
        recon_wb = iradon(sinogram_np, theta=angles, circle=True) / 255.
        unet = NestedUNet(nb_filter=(64, 128, 256, 512, 1024)).cuda()
        unet = unet.eval()
        unet.load_state_dict(torch.load(os.path.join(model_dir, "Denoise(sharp).pth")))
        size = sinogram_np.shape
        if size[0] == 128 or size[0] == 256 or size[0] == 512 or size[0] == 1024:
            pass
        else:
            if size[0] < 128:
                topad = 128
            elif size[0] < 256:
                topad = 256
            elif size[0] < 512:
                topad = 512
            elif size[0] < 1024:
                topad = 1024
            padleft = np.zeros((topad, (topad - size[0]) // 2))
            padright = np.zeros((topad, int(topad - size[0] - padleft.shape[1])))
            padup = np.zeros(((topad - size[0]) // 2, size[0]))
            paddown = np.zeros((int(topad - size[0] - padup.shape[0]), size[0]))
            recon_wb = np.concatenate((padup, recon_wb, paddown), axis=0)
            recon_wb = np.concatenate((padleft, recon_wb, padright), axis=1)
        transform = transforms.Compose([transforms.ToTensor()])
        ori_tensor = transform(recon_wb.astype(np.float32)).cuda()
        ori_tensor = torch.unsqueeze(ori_tensor, 0)
        # output = ori_tensor
        with torch.no_grad():
            for _ in range(numIter):
                # for i in range(ori_tensor.shape[0]):
                #     ori_tensor[i]=(ori_tensor[i] - ori_tensor[i].min()) / (ori_tensor[i].max() - ori_tensor[i].min())
                output = unet(ori_tensor)
        if size[0] == 128 or size[0] == 256 or size[0] == 512 or size[0] == 1024:
            recon_np = output.cpu().numpy()[0, 0, :,:]
        else:
            recon_np = output.cpu().numpy()[0, 0, padup.shape[0]:padup.shape[0]+size[0], padleft.shape[1]:padleft.shape[1]+size[0]]
        # recon_np = iradon(sinogram_np, theta=angles, circle=True)  # iradon 0. 255.
        return recon_np

    def irdm_80(sinogram_np, angles,model_dir, numIter=1):
        # print("--start irdm---")
        recon_wb = iradon(sinogram_np, theta=angles, circle=True) / 255.
        unet = NestedUNet(nb_filter=(32,64,128,256,512)).cuda()
        unet = unet.eval()
        unet.load_state_dict(torch.load(os.path.join(model_dir, "Denoise(80).pth")))
        size = sinogram_np.shape
        if size[0] == 128 or size[0] == 256 or size[0] == 512 or size[0] == 1024:
            pass
        else:
            if size[0] < 128:
                topad = 128
            elif size[0] < 256:
                topad = 256
            elif size[0] < 512:
                topad = 512
            elif size[0] < 1024:
                topad = 1024
            padleft = np.zeros((topad, (topad - size[0]) // 2))
            padright = np.zeros((topad, int(topad - size[0] - padleft.shape[1])))
            padup = np.zeros(((topad - size[0]) // 2, size[0]))
            paddown = np.zeros((int(topad - size[0] - padup.shape[0]), size[0]))
            recon_wb = np.concatenate((padup, recon_wb, paddown), axis=0)
            recon_wb = np.concatenate((padleft, recon_wb, padright), axis=1)
        transform = transforms.Compose([transforms.ToTensor()])
        ori_tensor = transform(recon_wb.astype(np.float32)).cuda()
        ori_tensor = torch.unsqueeze(ori_tensor, 0)
        # output = ori_tensor
        with torch.no_grad():
            for _ in range(numIter):
                # for i in range(ori_tensor.shape[0]):
                #     ori_tensor[i]=(ori_tensor[i] - ori_tensor[i].min()) / (ori_tensor[i].max() - ori_tensor[i].min())
                output = unet(ori_tensor)
        if size[0] == 128 or size[0] == 256 or size[0] == 512 or size[0] == 1024:
            recon_np = output.cpu().numpy()[0, 0, :,:]
        else:
            recon_np = output.cpu().numpy()[0, 0,padup.shape[0]:padup.shape[0]+size[0], padleft.shape[1]:padleft.shape[1]+size[0]]
        # recon_np = iradon(sinogram_np, theta=angles, circle=True)  # iradon 0. 255.
        return recon_np

    @staticmethod
    def sirt_xin(sinogram_np, angles, numIter=10):
        # SIRT
        # print("--start sirt_xin---")
        recon_SIRT = iradon(sinogram_np, theta=angles, filter=None, circle=True)
        for i in range(0, numIter):
            reprojs = radon(recon_SIRT, theta=angles, circle=True)
            ratio = sinogram_np / reprojs
            ratio[np.isinf(ratio)] = 1e8
            ratio[np.isnan(ratio)] = 1
            timet = iradon(ratio, theta=angles, filter=None, circle=True)
            timet[np.isinf(timet)] = 1
            timet[np.isnan(timet)] = 1
            recon_SIRT = recon_SIRT * timet
            # print('SIRT  %.3g' % i)
            # plt.imshow(recon_SIRT, cmap=plt.cm.Greys_r)
            # plt.show()
        return recon_SIRT

    @staticmethod
    def sart_cuda(sinogram_np, angles, step=100):
        # print("--start sart_cuda---")
        angles = [np.pi * ang / 180. for ang in angles]
        rec = tomopy.recon(sinogram_np.transpose(1, 0)[:,np.newaxis,:], angles, algorithm=tomopy.astra,
                           options={'method':'SART_CUDA',
                          'num_iter':step,
                          'proj_type':'cuda',
                          'extra_options':{'MinConstraint':0}
                          })
        return rec

    @staticmethod
    def sirt_cuda(sinogram_np, angles, numIter=100):
        # SIRT_CUDA
        # print("--start sirt_cuda---")
        angles = [np.pi * ang / 180. for ang in angles]
        rec = tomopy.recon(sinogram_np.transpose(1, 0)[:, np.newaxis, :], angles, algorithm=tomopy.astra,
                           options={'method': 'SIRT_CUDA',
                                    'num_iter': numIter,
                                    'proj_type': 'cuda',
                                    'extra_options': {'MinConstraint': 0}
                                    })
        return rec

    # @staticmethod
    # def sart_tvm(tils, angles, sart_iters=10, tv_w=0.1, tv_maxit=1000, tv_eps=1e-5, relaxation=0.3):
    #     image = None
    #     for _ in range(sart_iters):
    #         image = iradon_sart(tils, angles, image=image, relaxation=relaxation)
    #         image = denoise_tv_bregman(image, tv_w, eps=tv_eps, max_iter=tv_maxit, isotropic=False)
    #     image = iradon_sart(tils, angles, image=image, relaxation=relaxation)
    #     return image
    @staticmethod
    def sart_tvm(tils, angles, tau=0.2, round_num=5, tvm_iteration=200, relaxation=0.3):
        # print("--start sart_tvm---")
        denoise_TVM = None
        tils = tils.astype(np.float)
        for epoch in range(round_num):
            # SART
            if epoch >= 0:
                angle = [np.pi * ang / 180. for ang in angles]
                # recon_SART = tomopy.recon(tils.transpose(1, 0)[:, np.newaxis, :], angle, algorithm=tomopy.astra,
                #                    options={'method': 'SART_CUDA',
                #                             'num_iter': 100,
                #                             'proj_type': 'cuda',
                #                             'extra_options': {'MinConstraint': 0}
                #                             })[0]
                recon_SART = iradon_sart(tils, theta=angles, image=None, relaxation=relaxation)
            else:
                recon_SART = iradon_sart(tils, theta=angles, image=denoise_TVM, relaxation=relaxation)
            unit_tvm = np.zeros(recon_SART.shape)
            image = rof.denoise(recon_SART, unit_tvm, tau=tau, Ngrad=tvm_iteration)[0]
        return image

    @classmethod
    def fusion_3d(cls, sinogram_np, angles, model3d_dir, numIter=1):
        # print("--start irdm---")
        wbp_input = Normalize(cls.wbp(sinogram_np, angles).astype('float16')[:, :, np.newaxis])
        irdm_input = Normalize(cls.irdm(sinogram_np, angles, os.path.dirname(model3d_dir)).astype('float16')[:, :, np.newaxis])
        irdm_80_input = Normalize(cls.irdm_80(sinogram_np, angles, os.path.dirname(model3d_dir)).astype('float16')[:, :, np.newaxis])
        sirt_xin_input = Normalize(cls.sirt_xin(sinogram_np, angles).astype('float16')[:, :, np.newaxis])
        sart_tvm_input = Normalize(cls.sart_tvm(sinogram_np, angles).astype('float16')[:, :, np.newaxis])
        sirt_cuda_input = Normalize(cls.sirt_cuda(sinogram_np, angles)[0].astype('float16')[:, :, np.newaxis])
        sart_cuda_input = Normalize(cls.sart_cuda(sinogram_np, angles)[0].astype('float16')[:, :, np.newaxis])
        img_input = np.concatenate([wbp_input, irdm_input, irdm_80_input, sirt_xin_input, sart_tvm_input, \
                                    sirt_cuda_input, sart_cuda_input], 2)  # irdm,
        size = img_input.shape
        if size[0] == 128 or size[0] == 256 or size[0] == 512 or size[0] == 1024:
            pass
        else:
            if size[0] < 128:
                topad = 128
            elif size[0] < 256:
                topad = 256
            elif size[0] < 512:
                topad = 512
            elif size[0] < 1024:
                topad = 1024
            padleft = np.zeros((topad, (topad - size[0]) // 2, 7))
            padright = np.zeros((topad, int(topad - size[0] - padleft.shape[1]), 7))
            padup = np.zeros(((topad - size[0]) // 2, size[0], 7))
            paddown = np.zeros((int(topad - size[0] - padup.shape[0]), size[0], 7))
            img_input = np.concatenate((padup, img_input, paddown), axis=0)
            img_input = np.concatenate((padleft, img_input, padright), axis=1)
        transform = transforms.Compose([transforms.ToTensor()])
        img_input = transform(img_input)
        start_time = time.time()
        # model3d_dir = "../../model/Deepfusion/models_3dGAN_nodog/Deepfusion_4.pth"
        m = unet_standard_3d.NestedUNet(nb_filter=(32, 64, 128, 256, 512)).cuda()
        if torch.cuda.device_count() > 1:
            m = torch.nn.DataParallel(m)
        m = m.eval()
        m.load_state_dict(torch.load(model3d_dir))
        ori_tensor = torch.unsqueeze(img_input.type(torch.cuda.FloatTensor).cuda(), 0)
        output = ori_tensor
        with torch.no_grad():
            for _ in range(numIter):
                output = m(output)
        y = (output.data).cpu().numpy()[0, 0, :, :]
        end_time = time.time()
        # print("The time cost of %s is: %s" % ("fusion_3d", end_time - start_time))

        return y

    @classmethod
    def fusion_3dGAN(cls, sinogram_np, angles, model3dGAN_dir, numIter=1):
        # print("--start irdm---")
        wbp_input = Normalize(cls.wbp(sinogram_np, angles).astype('float16')[:, :, np.newaxis])
        irdm_input = Normalize(
            cls.irdm(sinogram_np, angles, os.path.dirname(model3dGAN_dir)).astype('float16')[:, :, np.newaxis])
        irdm_80_input = Normalize(
            cls.irdm_80(sinogram_np, angles, os.path.dirname(model3dGAN_dir)).astype('float16')[:, :, np.newaxis])
        sirt_xin_input = Normalize(cls.sirt_xin(sinogram_np, angles).astype('float16')[:, :, np.newaxis])
        sart_tvm_input = Normalize(cls.sart_tvm(sinogram_np, angles).astype('float16')[:, :, np.newaxis])
        sirt_cuda_input = Normalize(cls.sirt_cuda(sinogram_np, angles)[0].astype('float16')[:, :, np.newaxis])
        sart_cuda_input = Normalize(cls.sart_cuda(sinogram_np, angles)[0].astype('float16')[:, :, np.newaxis])
        img_input = np.concatenate([wbp_input, irdm_input, irdm_80_input, sirt_xin_input, sart_tvm_input, \
                                    sirt_cuda_input, sart_cuda_input], 2)  # irdm,
        size = img_input.shape
        if size[0] == 128 or size[0] == 256 or size[0] == 512 or size[0] == 1024:
            pass
        else:
            if size[0] < 128:
                topad = 128
            elif size[0] < 256:
                topad = 256
            elif size[0] < 512:
                topad = 512
            elif size[0] < 1024:
                topad = 1024
            padleft = np.zeros((topad, (topad - size[0]) // 2, 7))
            padright = np.zeros((topad, int(topad - size[0] - padleft.shape[1]), 7))
            padup = np.zeros(((topad - size[0]) // 2, size[0], 7))
            paddown = np.zeros((int(topad - size[0] - padup.shape[0]), size[0], 7))
            img_input = np.concatenate((padup, img_input, paddown), axis=0)
            img_input = np.concatenate((padleft, img_input, padright), axis=1)
        transform = transforms.Compose([transforms.ToTensor()])
        img_input = transform(img_input)
        start_time = time.time()
        # model3dGAN_dir = "../../model/Deepfusion/models_3dGAN_nodog/Deepfusion_4.pth"
        m = unet_standard_3d.NestedUNet(nb_filter=(32, 64, 128, 256, 512)).cuda()
        if torch.cuda.device_count() > 1:
            m = torch.nn.DataParallel(m)
        m = m.eval()
        m.load_state_dict(torch.load(model3dGAN_dir))
        ori_tensor = torch.unsqueeze(img_input.type(torch.cuda.FloatTensor).cuda(), 0)
        output = ori_tensor
        with torch.no_grad():
            for _ in range(numIter):
                output = m(output)
        y = (output.data).cpu().numpy()[0, 0, :, :]
        end_time = time.time()
        # print("The time cost of %s is: %s" % ("fusion_3dGAN", end_time - start_time))

        return y

    @staticmethod
    def add_gaussian_noise(img, mean=0, vars=(0.00008, 0.0008)):
        img_np = np.asarray(img)
        img_min, img_max = img_np.min(), img_np.max()
        img_np = (img_np - img_min) / (img_max - img_min + 1e-6)
        var = np.random.random() * (vars[1] - vars[0]) + vars[0]
        # img_np = np.asarray(img) / 255.0
        img_noised_np = skimage.util.random_noise(img_np, mode="gaussian", mean=mean, var=var)
        img_noised_np = img_noised_np * (img_max - img_min) + img_min
        return img_noised_np

    @staticmethod
    def add_poisson_noise(img):
        img_np = np.asarray(img)
        img_min, img_max = img_np.min(), img_np.max()
        img_np = (img_np - img_min) / (img_max - img_min + 1e-6)

        img_noised_np = skimage.util.random_noise(img_np, mode="poisson")
        img_noised_np = img_noised_np * (img_max - img_min) + img_min
        return img_noised_np

    @staticmethod
    def add_sp_noise(img):
        img_np = np.asarray(img)
        img_min, img_max = img_np.min(), img_np.max()
        img_np = (img_np - img_min) / (img_max - img_min + 1e-6)

        img_noised_np = skimage.util.random_noise(img_np, mode="s&p", amount=0.01)
        img_noised_np = img_noised_np * (img_max - img_min) + img_min
        return img_noised_np

if __name__ == '__main__':
    # =====================original===================================
    img_filenames = os.listdir(original_imgs_dir)
    # ======================noised=====================================
    for name in tqdm(img_filenames[:1]):
        img_path = os.path.join(original_imgs_dir, name)
        with Image.open(img_path) as img:
            img = img.convert("L")
            img_np = np.array(img)
            delta = 0.703125
            gap_degree = 60.
            angles = np.concatenate(
                (np.arange(0, 90 - gap_degree / 2., delta), np.arange(90 + gap_degree / 2., 180, delta)))
            # angles = np.arange(0, 180, delta)

            sinogram = radon(img_np,
                             theta=angles,
                             circle=True)  # radon 0.255
            wbp_np = Recon.wbp(sinogram, angles=angles)  # iradon 0. 255.
            irdm_80_np = Recon.irdm_80(sinogram, angles=angles)  # iradon 0. 255.
            irdm_np = Recon.irdm(sinogram, angles=angles)  # iradon 0. 255.
            sart_tvm_np = Recon.sart_tvm(sinogram, angles=angles)  # iradon 0. 255.
            sirt_xin_np = Recon.sirt_xin(sinogram, angles=angles)  # iradon 0. 255.
            sart_cuda_np = Recon.sart_cuda(sinogram, angles=angles)  # iradon 0. 255.
            sirt_cuda_np = Recon.sirt_cuda(sinogram, angles=angles)  # iradon 0. 255.
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4*4,4*2))
            plt.subplot(241)
            plt.title("origin")
            plt.imshow(img_np.astype(np.float), cmap="gray")
            plt.subplot(242)
            plt.title("irdm")
            plt.imshow(irdm_np.astype(np.float), cmap="gray")
            plt.subplot(243)
            plt.title("sart_tvm")
            plt.imshow(sart_tvm_np.astype(np.float), cmap="gray")
            plt.subplot(244)
            plt.title("sirt_xin")
            plt.imshow(sirt_xin_np.astype(np.float), cmap="gray")
            plt.subplot(245)
            plt.title("sart_cuda")
            plt.imshow(sart_cuda_np[0].astype(np.float), cmap="gray")
            plt.subplot(246)
            plt.title("sirt_cuda")
            plt.imshow(sirt_cuda_np[0].astype(np.float), cmap="gray")
            plt.subplot(247)
            plt.title("wbp")
            plt.imshow(wbp_np.astype(np.float), cmap="gray")
            plt.subplot(248)
            plt.title("irdm_80_np")
            plt.imshow(irdm_80_np.astype(np.float), cmap="gray")
            plt.savefig("%s_recon.png" % name)

