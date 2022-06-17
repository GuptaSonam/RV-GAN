from __future__ import absolute_import

import torch
import torch.nn as nn
from model.predrnn_networks import RNN

import skvideo.io
import numpy as np
import os
from tqdm import tqdm
import argparse

def save_videos(path, vids, epoch, bs):

	for i in range(bs):
		v = vids[i].permute(0,2,3,1).cpu().numpy()
		v *= 255
		v = v.astype(np.uint8)
		skvideo.io.vwrite(os.path.join(path, "%d.mp4"%(epoch * bs + i)), v, outputdict={"-vcodec":"libx264"})



def main(args):
	print("generating videos for ", args.model_path)
	print("saving videos in ", args.gen_path)
	device = torch.device("cuda:0")

	G = RNN(args).to(device)
	G = nn.DataParallel(G)
	G.load_state_dict(torch.load(args.model_path))

	with torch.no_grad():
		G.eval()

		batch_size = args.batch_size
		n_epoch = args.n // batch_size + 1

		for epoch in tqdm(range(n_epoch)):

			bs = min(batch_size, args.n - epoch * batch_size)
			#zfg = torch.randn(bs, args.d_za, 1, 1, 1).to(device)
			z_a = torch.randn(bs, args.d_za, 1, 1, 1).to(device)
			#zm = torch.randn(bs, args.d_zm, 1, 1, 1).to(device)

			vid_fake = G(z_a)

			vid_fake = vid_fake.transpose(2,1) # bs x 16 x 3 x 64 x 64
			vid_fake = ((vid_fake - vid_fake.min()) / (vid_fake.max() - vid_fake.min())).data

			# save into videos
			save_videos(args.gen_path, vid_fake, epoch, bs)


	return


if __name__ == '__main__':

	# gen_path = '/data/stars/user/yaowang/exp/g3an/'

	# training params
	parser = argparse.ArgumentParser()
	parser.add_argument("--n", type=int, default=5000)
	parser.add_argument("--batch_size", type=int, default=12)
	parser.add_argument("--d_za", type=int, default=128)
	parser.add_argument("--d_zm", type=int, default=10)
	parser.add_argument("--model_path", type=str, default='/media/vplab/My Passport/sonam-arti/sonam-arti/wacv2022/exps/rnn_single_noise_no_blank/models/G_5000.pth')
	parser.add_argument("--gen_path", type=str, default='/media/vplab/My Passport/sonam-arti/sonam-arti/extension_g3an_WZM/evaluation/RNN_single_noise_no_blank_5000')
	parser.add_argument('--total_length', type=int, default='16')
	parser.add_argument('--layer_norm', type=int, default='1')

	args = parser.parse_args()

	main(args)
