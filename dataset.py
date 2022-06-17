import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import glob

class WZM(Dataset):
	def __init__(self, data_path, transform=None):

		self.data_path = data_path
		self.step = [1,2]
		#self.step = 1
		self.vids = os.listdir(self.data_path)
		self.transform = transform

	def __getitem__(self, idx):
		
		video_path = os.path.join(self.data_path, self.vids[idx])
		frames = sorted(glob.glob(video_path + '/*.png'))
		nframes = len(frames)
		#print(nframes)
		step = random.sample(self.step, 1)[0]
		#step = self.step                  
		if nframes<32:
			step = 1   
		start_idx = random.randint(0, nframes-16 * step)
		vid = [Image.open(frames[start_idx + i * step]).convert('RGB') for i in range(16)]			

		if self.transform is not None:
			vid = self.transform(vid)

		return vid

	def __len__(self):

		return len(self.vids)


if __name__ == '__main__':
	
	data_path = '/data/stars/user/yaowang/data/UVA/crop_faces/data/'
	
	dataset = WZM(data_path)
	for i in range(len(dataset)):
		vid = dataset.__getitem__(i)
		print(len(vid))
