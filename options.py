# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "oxford_all"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="oxford",
                                 choices=["oxford"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw Oxford png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=512)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--no_percep",
                                 help="if set, dont use perceptual loss",
                                 action="store_true")
        self.parser.add_argument("--only_im2im",
                                 help="if set, only train image to image task",
                                 action="store_true")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--load_pseudo_model",
                                 type=str,
                                 help="name of pseudo model to load")
        self.parser.add_argument("--load_day_dec",
                                 type=str,
                                 help="name of day decoder to load")
        self.parser.add_argument("--load_night_dec",
                                 type=str,
                                 help="name of night decoder to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 default=1,
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="night_val_411",
                                 choices=["night_val_411", "day_val_451"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
	   
	   # CYCLEGAN OPTIONS
	   # basic define
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
        # network structure define
        self.parser.add_argument('--image_nc', type=int, default=3,
                                 help='# of input image channels')
        self.parser.add_argument('--label_nc', type=int, default=1,
                                 help='# of output label channels')
        self.parser.add_argument('--ngf', type=int, default=64,
                                 help='# of encoder filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64,
                                 help='# of discriminator filter in first conv layer')
        self.parser.add_argument('--image_feature', type=int, default=512,
                                 help='the max channels for image features')
        self.parser.add_argument('--num_D', type=int, default=1,
                                 help='# of number of the discriminator')
        self.parser.add_argument('--transform_layers', type=int, default=9,
                                 help='# of number of the down sample layers for transform network')
        self.parser.add_argument('--task_layers', type=int, default=4,
                                 help='# of number of the down sample layers for task network')
        self.parser.add_argument('--image_D_layers', type=int, default=3,
                                 help='# of number of the down layers for image discriminator')
        self.parser.add_argument('--feature_D_layers', type=int, default=2,
                                 help='# of number of the layers for features discriminator')
        self.parser.add_argument('--task_model_type', type=str, default='UNet',
                                 help='select model for task network [UNet] |[ResNet]')
        self.parser.add_argument('--trans_model_type', type=str, default='ResNet',
                                 help='select model for transform network [UNet] |[ResNet]')
        self.parser.add_argument('--norm', type=str, default='batch',
                                 help='batch normalization or instance normalization')
        self.parser.add_argument('--activation', type=str, default='PReLU',
                                 help='ReLu, LeakyReLU, PReLU, or SELU')
        self.parser.add_argument('--init_type', type=str, default='kaiming',
                                 help='network initialization [normal|xavier|kaiming]')
        self.parser.add_argument('--drop_rate', type=float, default=0,
                                 help='# of drop rate')
        self.parser.add_argument('--U_weight', type=float, default=0.1,
                                 help='weight for Unet')
        self.parser.add_argument('--niter', type=int, default=6,
                                 help='# of iter with initial learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=4,
                                 help='# of iter to decay learning rate to zero')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--transform_epoch', type=int, default=0,
                                 help='# of epoch for transform learning')
        self.parser.add_argument('--task_epoch', type=int, default=0,
                                 help='# of epoch for task learning')
        # learning rate and loss weight
        self.parser.add_argument('--lr_policy', type=str, default='lambda',
                                 help='learning rate policy[lambda|step|plateau]')
        self.parser.add_argument('--lr_task', type=float, default=1e-4,
                                 help='initial learning rate for adam')
        self.parser.add_argument('--lr_trans', type=float, default=5e-5,
                                 help='initial learning rate for discriminator')
        self.parser.add_argument('--lambda_rec_img', type=float, default=100.0,
                                 help='weight for image reconstruction loss')
        self.parser.add_argument('--lambda_gan_img', type=float, default=1.0,
                                 help='weight for image GAN loss')
        self.parser.add_argument('--lambda_gan_feature', type=float, default=0.1,
                                 help='weight for feature GAN loss')
        self.parser.add_argument('--lambda_rec_lab', type=float, default=100.0,
                                 help='weight for task loss')
        self.parser.add_argument('--lambda_smooth', type=float, default=0.1,
                                 help='weight for depth smooth loss')
        
        self.parser.add_argument('--separate', action='store_true',
                                 help='transform and task network training end-to-end or separate')
        self.parser.add_argument('--pool_size', type=int, default=20,
                                 help='the size of image buffer that stores previously generated images')
        self.isTrain = True

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
