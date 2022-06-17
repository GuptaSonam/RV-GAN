__author__ = 'yunbo'

import torch
import torch.nn as nn
from layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
import torch.nn.utils.spectral_norm as spectralnorm
from torch.nn import init

class RNN(nn.Module):
    def __init__(self, configs):
        super(RNN, self).__init__()

        self.configs = configs
        #self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.frame_channel = 3
        num_layers = 5
        num_hidden = [512, 512, 256, 128, 64]
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        self.batch_size = configs.batch_size 
        width = [4,8,16,32,64]
        self.width = width
        filter_size = 4
        stride = [1,2,2,2,2]
        padding = [0,1,1,1,1]
        
        #width = configs.img_width // configs.patch_size
        #self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = configs.d_za  if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width[i], filter_size,
                                       stride[i], padding[i], configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
         
        self.conv_last = nn.Sequential(nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=3, stride=1, padding=1, bias=False),
			            nn.Tanh()
		)

    def forward(self, z_a):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        #frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        #mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        #batch = frames.shape[0]
        batch = z_a.size(0)
        next_frames = []
        h_t = []
        c_t = []
        #print('h_t_shape')
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.width[i], self.width[i]]).cuda()
            #print(zeros.shape)
            c_t.append(zeros)
            h_t.append(zeros)
        #print('done')
        #memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length):
            #net = mask_true[:, t] * frames[:, t] + \
            #              (1 - mask_true[:, t]) * x_gen
            #print("t ", t)
            h_t[0], c_t[0] = self.cell_list[0](z_a[:,:,0,:,:], h_t[0], c_t[0]) 
            
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        #print('next_frames ', next_frames[0].shape)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 2, 0,3,4).contiguous()
        #loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames


class VideoDiscriminator(nn.Module):
	def __init__(self, ch=64):
		super(VideoDiscriminator, self).__init__()

		self.net = nn.Sequential(
			spectralnorm(nn.Conv3d(3,	ch, (1,4,4), (1,2,2), (0,1,1))),
			nn.LeakyReLU(0.2, inplace=True),
			spectralnorm(nn.Conv3d(ch, ch, (4,1,1), 1, 0)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch, ch*2, (1,4,4), (1,2,2), (0,1,1))),
			nn.LeakyReLU(0.2, inplace=True),
			spectralnorm(nn.Conv3d(ch*2, ch*2, (4,1,1), 1, 0)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch*2, ch*4, (1,4,4), (1,2,2), (0,1,1))),
			nn.LeakyReLU(0.2, inplace=True),
			spectralnorm(nn.Conv3d(ch*4, ch*4, (4,1,1), 1, 0)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch*4, ch*8, (1,4,4), (1,2,2), (0,1,1))),
			nn.LeakyReLU(0.2, inplace=True),
			spectralnorm(nn.Conv3d(ch*8, ch*8, (4,1,1), 1, 0)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch*8, ch*8, (1,4,4), 1, 0)),
			nn.LeakyReLU(0.2, inplace=True),
			spectralnorm(nn.Conv3d(ch*8,	1, (4,1,1), 1, 0))
		)

		self.init_weights()

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv3d):
				init.normal_(module.weight, 0, 0.02)

	def forward(self, x):

		out = self.net(x)

		return out.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)


class ImageDiscriminator(nn.Module):
	def __init__(self, ch=64):
		super(ImageDiscriminator, self).__init__()

		self.net = nn.Sequential(
			spectralnorm(nn.Conv2d(3, ch, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch, ch*2, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch*2, ch*4, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch*4, ch*8, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch*8, 1, 4, 1, 0)),
		)

		self.init_weights()

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				init.normal_(module.weight, 0, 0.02)

	def forward(self, x):

		out = self.net(x)

		return out.squeeze(-1).squeeze(-1).squeeze(-1)

