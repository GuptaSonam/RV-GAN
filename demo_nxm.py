from __future__ import absolute_import

import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from model.predrnn_networks import RNN
import cfg
import skvideo.io
import numpy as np
import os
from PIL import Image as im
import copy

# def stitch(v):
# 	new_v = 255 * np.ones(v.shape[0], v.shape[1]*v.shape[2]+2*(v.shape[1]-1), v.shape[3])
# 	for 
def save_video_frames(path, vids, n_zb):
	for i in range(n_zb): # foreground loop
		v = vids[i].permute(2,0,3,1).cpu().numpy()
		v *= 255
		v = v.astype(np.uint8)							#(16,64,64,3)
		v = v.reshape(v.shape[0], v.shape[1]*v.shape[2], v.shape[3])
		filepath = os.path.join(path, "vid_%d.png"%(i)) #'/vid-%d.png' %( t))       
		img = im.fromarray(v)
		img.save(filepath) 
	return

def save_video_frames_(path, vids, n_za):
	for i in range(n_za): 			
		v = vids[n_za*i].permute(0,2,3,1).cpu().numpy()
		v *= 255
		v = v.astype(np.uint8)							#(16,64,64,3)
		#skvideo.io.vwrite(os.path.join(path, "%d_%d.mp4"%(i, j)), v, outputdict={"-vcodec":"libx264"})
		os.mkdir(os.path.join(path, "%d_%d"%(i, j)))
		for t in range(v.shape[0]):
			filepath = os.path.join(os.path.join(path, "%d"%(i)) + '/vid-%d.png' %(t))        
			img = im.fromarray(v[t,:,:,:])
			img.save(filepath) 
	return

def save_videos(path, vids, n_za):
	for i in range(n_za): # appearance loop
		v = vids[n_za*i + j].permute(0,2,3,1).cpu().numpy()
		v *= 255
		v = v.astype(np.uint8)
		skvideo.io.vwrite(os.path.join(path, "%d.mp4"%(i)), v, outputdict={"-vcodec":"libx264"})

	return

def latent_space_manipulation(path, log_path, vid_path, za, G):
	# write into tensorboard
	log_path = os.path.join(log_path, path)
	vid_path = os.path.join(vid_path, path)
	
	if not os.path.exists(log_path) and not os.path.exists(vid_path):
		os.makedirs(log_path)
		os.makedirs(vid_path)
		
	n_za = za.size(0)
	print(n_za)

	#za = za.unsqueeze(1).repeat(1, n_zm, 1, 1, 1, 1).contiguous().view(n_za*n_zm, -1, 1, 1, 1)
	
	vid_fake = G(za)
	print('vid_fake ', vid_fake.shape)
	vid_fake = vid_fake.transpose(2,1) # bs x 16 x 3 x 64 x 64
	vid_fake = ((vid_fake - vid_fake.min()) / (vid_fake.max() - vid_fake.min())).data

	# save into videos
	print('==> saving videos...')
	save_video_frames(vid_path, vid_fake, n_za)
	

def main():

	args = cfg.parse_args()

	# write into tensorboard
	log_path = os.path.join(args.demo_path, args.demo_name + '/log')
	vid_path = os.path.join(args.demo_path, args.demo_name + '/vids')
	
	if not os.path.exists(log_path) and not os.path.exists(vid_path):
		os.makedirs(log_path)
		os.makedirs(vid_path)
		
	device = torch.device("cuda:0")

	G = RNN(args).to(device)
	G = nn.DataParallel(G)
	G.load_state_dict(torch.load(args.model_path))

	with torch.no_grad():
		G.eval()
		seed = 123
		torch.manual_seed(seed)
		
		za_orig = torch.randn(args.batch_size, args.d_za, 16, 1, 1).to(device)
		#vid = G(za_orig)
		# save original video
		latent_space_manipulation('base_' + str(seed), log_path, vid_path, za_orig, G)
		
	return


if __name__ == '__main__':

	main()