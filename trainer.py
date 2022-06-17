import os.path
import datetime
import cv2
import numpy as np
from skimage.measure import compare_ssim
#from utils import preprocess, metrics
#import lpips
import torch
import random
from PIL import Image as im

#loss_fn_alex = lpips.LPIPS(net='alex')
def save_video_frames_(path, vids, n_za=8):
	print(vids.shape)
	for i in range(n_za): 			
		v = vids[n_za*i].permute(1,2,3,0).cpu().numpy()
		v *= 255
		print(v.shape)
		v = v.astype(np.uint8)							#(16,64,64,3)
		#skvideo.io.vwrite(os.path.join(path, "%d_%d.mp4"%(i, j)), v, outputdict={"-vcodec":"libx264"})
		os.mkdir(os.path.join(path, "%d"%(i)))
		for t in range(v.shape[0]):
			filepath = os.path.join(os.path.join(path, "%d"%(i)) + '/vid-%d.png' %(t))        
			img = im.fromarray(v[t,:,:,:])
			img.save(filepath) 
	return

def train(args, epoch, G, VD, ID, optimizer_G, optimizer_VD, optimizer_ID, criterion, dataloader, device):
    #train mode
    G.train()
    VD.train()
    ID.train()
    
    for i, x in enumerate(dataloader):
        global_steps = epoch * len(dataloader) + i
        bs = x.size(0)
        real_vid = x.to(device)
        real_img = real_vid[:,:,random.randint(0, x.size(2)-1), :, :]
		#shuffled_vid = get_shuffle_video(x).to(device)


		#################### train D ##################
        optimizer_VD.zero_grad()
        optimizer_ID.zero_grad()

        VD_real = VD(real_vid)
		#VD_real_shuffle = VD(shuffled_vid)
        ID_real = ID(real_img)
        
        za = torch.randn(bs, args.d_za, 1, 1, 1).to(device).repeat(1,1,16,1,1)
        #zm = torch.randn(bs, args.d_zm, 16, 1, 1).to(device)

        # za = torch.cat([za,zm],1) 
        fake_vid = G(za)
        fake_img = fake_vid[:,:, random.randint(0, x.size(2)-1),:,:]
        VD_fake = VD(fake_vid.detach())
        ID_fake = ID(fake_img.detach())
        
        y_real = torch.ones(VD_real.size()).to(device)
        y_fake = torch.zeros(VD_fake.size()).to(device)
        
        errVD = criterion(VD_real, y_real) + criterion(VD_fake, y_fake) 
        errID = criterion(ID_real, y_real) + criterion(ID_fake, y_fake)
		
        errVD.backward()
        optimizer_VD.step()
        
        errID.backward()
        optimizer_ID.step()

		################## train G ###################
        optimizer_G.zero_grad()
        
        VG_fake = VD(fake_vid)
        IG_fake = ID(fake_img)
        
        errVG = criterion(VG_fake, y_real)
        errIG = criterion(IG_fake, y_real)
        errG = errVG + errIG
        
        errG.backward()
        optimizer_G.step()
		
        '''
		writer.add_scalar('G_vid_loss', errVG.item(), global_steps)
		writer.add_scalar('G_img_loss', errIG.item(), global_steps)
		writer.add_scalar('D_vid_loss', errVD.item(), global_steps)
		writer.add_scalar('D_img_loss', errID.item(), global_steps)
		writer.flush()
		'''	
        if global_steps % args.print_freq == 0:
            print("[Epoch %d/%d] [Iter %d/%d] [VD loss: %f] [VG loss: %f] [ID loss: %f] [IG loss: %f]"
			      %(epoch, args.max_epoch, i, global_steps, errVD.item(), errVG.item(), errID.item(), errIG.item()))

def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)

        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_gen_length = img_gen.shape[1]
        img_out = img_gen[:, -output_length:]

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # cal lpips
            img_x = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                ssim[i] += score

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(img_gen_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_gen[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print(lp[i])
