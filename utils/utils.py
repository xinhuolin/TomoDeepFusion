import torch
from torchvision.transforms import ToTensor
from PyQt5 import QtGui
import os
import numpy as np
from model_structure.Recon import Recon
from torchvision import transforms
from tqdm import tqdm
from skimage.transform import radon, iradon, iradon_sart

def GetIndexRangeOfBlk(height, width, blk_row, blk_col, blk_r, blk_c, over_lap = 0):
	blk_h_size = height//blk_row
	blk_w_size = width//blk_col

	if blk_r >= blk_row or blk_c >= blk_col:
		raise Exception("index is out of range...")

	upper_left_r = blk_r * blk_h_size
	upper_left_c = blk_c * blk_w_size
	ol_upper_left_r = max(upper_left_r - over_lap, 0)
	ol_upper_left_c = max(upper_left_c - over_lap, 0)

	if blk_r == (blk_row - 1):
		lower_right_r = height
		ol_lower_right_r = lower_right_r
	else:
		lower_right_r = upper_left_r + blk_h_size
		ol_lower_right_r = min(lower_right_r + over_lap, height)

	if blk_c == (blk_col - 1):
		lower_right_c = width
		ol_lower_right_c = lower_right_c
	else:
		lower_right_c = upper_left_c + blk_w_size
		ol_lower_right_c = min(lower_right_c + over_lap, width)

	return (upper_left_c, upper_left_r, lower_right_c, lower_right_r), (ol_upper_left_c, ol_upper_left_r, ol_lower_right_c, ol_lower_right_r)

def load_model(name, matdata, matangles, size, model_path, recon_path, cuda, sart_iter, sirt_iter, iter = 1):
	angle = matangles[0].astype(np.float)
	tomos = {}
	tomos_r = {}
	tomos_d = {}
	is_fusion = 0
	if os.path.basename(model_path) == "Denoise(sharp).pth":
		from model_structure.model import NestedUNet
		unet = NestedUNet(nb_filter=(64, 128, 256, 512, 1024))
	elif os.path.basename(model_path) == "Denoise(delta10).pth":
		from model_structure.model import NestedUNet
		unet = NestedUNet()
	elif os.path.basename(model_path) == "Denoise(IC).pth":
		from model_structure.model import NestedUNet
		unet = NestedUNet()
	elif os.path.basename(model_path) == "Denoise(80).pth":
		from model_structure.unet_standard import NestedUNet
		unet = NestedUNet(nb_filter=(32, 64, 128, 256, 512))
	elif os.path.basename(model_path) == "Deepfusion(3d).pth":
		is_fusion = 1
	elif os.path.basename(model_path) == "Deepfusion(3dGAN).pth":
		is_fusion = 2
	else:
		raise ValueError
	size = list(size)
	size[0], size[1] = size[1], size[0]
	tomos[name] = np.zeros((size[0], size[1], size[1]))
	tomos_r[name] = np.zeros((size[0], size[1], size[1]))
	for j in tqdm(range(size[0])):
		sinogram = np.array(matdata[name][:,j,:]).astype(np.float)
		torch.cuda.empty_cache()
		if is_fusion==1:
			tomogram = Recon.fusion_3d(sinogram, angle, model_path)
			if recon_path == "wbp":
				tomogram_recon = iradon(sinogram, theta=angle)
			elif recon_path == "sart":
				for e in range(sart_iter):
					if e==0:
						tomogram_recon = iradon_sart(sinogram, theta=angle, image=None, relaxation=0.3)
					else:
						tomogram_recon = iradon_sart(sinogram, theta=angle, image=tomogram_recon, relaxation=0.3)
			elif recon_path == "sirt":
				tomogram_recon = Recon.sirt_xin(sinogram, angle, sirt_iter)  # iradon 0. 255.
			else:
				raise ValueError
			tomos_r[name][j, :, :] = tomogram_recon.copy()
		elif is_fusion==2:
			tomogram = Recon.fusion_3dGAN(sinogram, angle, model_path)
			if recon_path == "wbp":
				tomogram_recon = iradon(sinogram, theta=angle)
			elif recon_path == "sart":
				for e in range(sart_iter):
					if e==0:
						tomogram_recon = iradon_sart(sinogram, theta=angle, image=None, relaxation=0.3)
					else:
						tomogram_recon = iradon_sart(sinogram, theta=angle, image=tomogram_recon, relaxation=0.3)
			elif recon_path == "sirt":
				tomogram_recon = Recon.sirt_xin(sinogram, angle, sirt_iter)  # iradon 0. 255.
			else:
				raise ValueError
			tomos_r[name][j, :, :] = tomogram_recon.copy()
		else:
			if recon_path == "wbp":
				tomogram = iradon(sinogram, theta=angle)
			elif recon_path == "sart":
				for e in range(sart_iter):
					if e==0:
						tomogram = iradon_sart(sinogram, theta=angle, image=None, relaxation=0.3)
					else:
						tomogram = iradon_sart(sinogram, theta=angle, image=tomogram, relaxation=0.3)
			elif recon_path == "sirt":
				tomogram = Recon.sirt_xin(sinogram, angle, sirt_iter)  # iradon 0. 255.
			else:
				raise ValueError

		tomos[name][j, :, :] = tomogram.copy()
	if is_fusion>0:
		tomos_d[name] = tomos[name]
		tomos[name] = tomos_r[name]
	else:
		topad = size[1]
		if size[1]==128 or size[1]==256 or size[1]==512 or size[1]==1024:
			pass
		else:
			if size[1]<128:
				topad = 128
			elif size[1] < 256:
				topad = 256
			elif size[1] < 512:
				topad = 512
			elif size[1] < 1024:
				topad = 1024
			padleft = np.zeros((size[0],topad,(topad-size[1])//2))
			padright = np.zeros((size[0],topad,int(topad-size[1]-padleft.shape[2])))
			padup = np.zeros((size[0], (topad - size[1]) // 2, size[1]))
			paddown = np.zeros((size[0], int(topad - size[1] - padup.shape[1]), size[1]))
			tomos[name] = np.concatenate((padup,tomos[name],paddown),axis=1)
			tomos[name] = np.concatenate((padleft,tomos[name],padright),axis=2)

		transform = transforms.Compose([transforms.ToTensor()])
		img = transform(tomos[name].transpose((1, 2, 0)))




		if cuda:
			unet = unet.cuda()
			# if torch.cuda.device_count() > 1:
			# 	unet = torch.nn.DataParallel(unet)
			unet.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path).items()})
			ori_tensor = torch.unsqueeze(img.type(torch.cuda.FloatTensor).cuda(), 1)
		else:
			unet.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path).items()})
			ori_tensor = torch.unsqueeze(img.type(torch.FloatTensor), 1)

		tomos_d[name] = np.zeros((size[0], topad, topad))
		with torch.no_grad():
			for idx in tqdm(range(ori_tensor.shape[0])):
				for j in range(iter):
					if j==0:
						rec = unet(ori_tensor[idx:idx+1,:,:,:])
					else:
						rec = unet(rec)
				rec = rec.cpu().numpy()[:, 0, :, :]
				tomos_d[name][idx] = rec


	return tomos_d, tomos


def PIL2Pixmap(im):
    """Convert PIL image to QImage """
    if im.mode == "RGB":
        pass
    elif im.mode == "L":
        im = im.convert("RGBA")
    data = im.convert("RGBA").tobytes("raw", "RGBA")
    qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_ARGB32)
    pixmap = QtGui.QPixmap.fromImage(qim)
    return pixmap

def map01(mat):
    return (mat - mat.min())/(mat.max() - mat.min())
