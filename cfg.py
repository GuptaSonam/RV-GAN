import argparse


def parse_args():

	parser = argparse.ArgumentParser('g3an training config')

	# train
	parser.add_argument('--max_epoch', type=int, default=10001, help='number of epochs of training')
	parser.add_argument('--batch_size', type=int, default=8, help='size of the batch')
	parser.add_argument('--g_lr', type=float, default=2e-4, help='learning rate of generator')
	parser.add_argument('--d_lr', type=float, default=2e-4, help='learning rate of discriminator')
	parser.add_argument('--d_za', type=int, default=128, help='appearance dim')
	parser.add_argument('--d_zm', type=int, default=10, help='motion dim')
	parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
	parser.add_argument('--ch_g', type=int, default=64, help='base channels of generator')
	parser.add_argument('--ch_d', type=int, default=64, help='base channels of discriminator')
	parser.add_argument('--g_mode', type=str, default='1p2d', choices=['1p2d', '2p1d', '3d'], help='generator operation mode')
	parser.add_argument('--img_size', type=int, default=64, help='input image size')
	parser.add_argument('--img_width', type=int, default=85, help='input image size')
	parser.add_argument('--img_height', type=int, default=96, help='input image size')
	parser.add_argument('--dataset', type=str, default='uva', choices=['uva'], help='dataset choice')
	parser.add_argument('--val_freq', type=int, default=25, help='validation frequence')
	parser.add_argument('--print_freq', type=int, default=100, help='log frequence')
	parser.add_argument('--save_freq', type=int, default=100, help='model save frequence')
	parser.add_argument('--exp_name', type=str, default='rnn_single_noise_no_blank')
	parser.add_argument('--save_path', type=str, default='/media/anydesk/My Passport/sonam-arti/sonam-arti/wacv2022/exps', help='model and log save path')
	parser.add_argument('--data_path', type=str, default='/media/anydesk/My Passport/sonam-arti/sonam-arti/all_dataset/wzm/preprocess_wzm_frame_all_no_blank', help='dataset path')
	parser.add_argument('--use_attention', action='store_true', default=False, help='whether to use attention')
	parser.add_argument('--random_seed', type=int, default='12345')
	parser.add_argument('--patch_size', type=int, default='8')
	parser.add_argument('--total_length', type=int, default='16')
	parser.add_argument('--layer_norm', type=int, default='1')

	# test
	parser.add_argument('--n', type=int, default=8, help='number of random generation')
	parser.add_argument('--n_za_test', type=int, default=8, help='number of foreground')
	parser.add_argument('--n_zm_test', type=int, default=3, help='number of motion')
	#parser.add_argument('--n_zb_test', type=int, default=2, help='number of background')
	parser.add_argument('--demo_name', type=str, default='v_5000', help='name of demo')
	parser.add_argument('--model_path', type=str, default='/media/anydesk/My Passport/sonam-arti/sonam-arti/extension_g3an_WZM/exps/rnn_single_noise_no_blank/models/G_5000.pth', help='pre-trained model path')
	parser.add_argument('--demo_path', type=str, default='/media/anydesk/My Passport/sonam-arti/sonam-arti/extension_g3an_WZM/exps/rnn_single_noise_no_blank/demos', help='demos save path')
	parser.add_argument('--pre_trained_model_path', type=str, default='')

	args = parser.parse_args()

	return args
