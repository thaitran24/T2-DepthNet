# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import itertools
import datasets
import networks
from IPython import embed
from PIL import Image
from util.image_pool import ImagePool
from util import task
from util.visualizer import Visualizer

import cv2
STEREO_SCALE_FACTOR = 5.4
DAY_PHASE = 1
NIGHT_PHASE = 2 

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.result_path = os.path.join(self.log_path, 'result.txt')

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.train_params = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        # Shared Encoder
        if not self.opt.only_im2im:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.train_params += list(self.models["encoder"].parameters())

            # Depth Decoder
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.train_params += list(self.models["depth"].parameters())

        if self.opt.load_pseudo_model and not self.opt.only_im2im:
            self.pseudo_models = {}
            # Pseudo Model Encoder and Decoder
            self.pseudo_models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.pseudo_models["encoder"].to(self.device)

            self.pseudo_models["depth"] = networks.DepthDecoder(
                self.pseudo_models["encoder"].num_ch_enc, self.opt.scales)
            self.pseudo_models["depth"].to(self.device)

            self.load_pseudo_model()

            for m in self.pseudo_models.values():
                m.eval()
        
        if self.use_pose_net and not self.opt.only_im2im:
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["pose_encoder"].to(self.device)
            self.train_params += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

            self.models["pose"].to(self.device)
            self.train_params += list(self.models["pose"].parameters())

        if not self.opt.only_im2im:
            # Depth optimizer
            self.model_optimizer = optim.Adam(self.train_params, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        
        # Image mapping model
        self.im2im_G = networks.define_G(self.opt.image_nc, self.opt.image_nc, self.opt.ngf, self.opt.transform_layers, self.opt.norm,
                                                  self.opt.activation, self.opt.trans_model_type, self.opt.init_type, self.opt.drop_rate,
                                                  False, self.opt.gpu_ids, self.opt.U_weight)
        self.optimizer_G = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.im2im_G.parameters())}],
                                            lr=self.opt.lr_trans, betas=(0.5, 0.9))
        
        # Discriminator
        self.im2im_D = networks.define_D(self.opt.image_nc, self.opt.ndf, self.opt.image_D_layers, self.opt.num_D, self.opt.norm,
                                              self.opt.activation, self.opt.init_type, self.opt.gpu_ids)

        if not self.opt.only_im2im:
            self.feat_D = networks.define_featureD(self.opt.image_feature, self.opt.feature_D_layers, self.opt.norm,
                                                   self.opt.activation, self.opt.init_type, self.opt.gpu_ids)
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.im2im_D.parameters()),
                                                                filter(lambda p: p.requires_grad, self.feat_D.parameters())),
                                                lr=self.opt.lr_trans, betas=(0.5, 0.9))
        else:
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.im2im_D.parameters())),
                                                lr=self.opt.lr_trans, betas=(0.5, 0.9))

        self.im2im_schedulers = []
        self.im2im_optimizers = [self.optimizer_D, self.optimizer_G]
        for optimizer in self.im2im_optimizers:
                self.im2im_schedulers.append(networks.get_scheduler(optimizer, self.opt))

        # Image pool
        self.fake_img_pool = ImagePool(self.opt.pool_size)

        # Loss function
        self.l1loss = torch.nn.L1Loss().cuda()
        self.mse = torch.nn.MSELoss().cuda()
        self.nonlinearity = torch.nn.ReLU()
        if not self.opt.no_percep:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

        # Load pretrained
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # Data
        datasets_dict = {"oxford": datasets.OxfordRawDataset, 
                         "oxford_pair": datasets.OxfordRawPairDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_day_filenames = readlines(fpath.format("train_day"))
        val_day_filenames = readlines(fpath.format("val_day"))
        train_night_filenames = readlines(fpath.format("train_night"))
        val_night_filenames = readlines(fpath.format("val_night"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_day_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_day_dataset = self.dataset(
            self.opt.data_path, train_day_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_day_loader = DataLoader(
            train_day_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        train_night_dataset = self.dataset(
            self.opt.data_path, train_night_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_night_loader = DataLoader(
            train_night_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        if not self.opt.only_im2im:
            val_day_dataset = self.dataset(
                self.opt.data_path, val_day_filenames, self.opt.height, self.opt.width,
                [0], 4, is_train=False, img_ext='.png')
            self.val_day_loader = DataLoader(
                val_day_dataset, self.opt.batch_size, False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            
            val_night_dataset = self.dataset(
                self.opt.data_path, val_night_filenames, self.opt.height, self.opt.width,
                [0], 4, is_train=False, img_ext='.png')
            self.val_night_loader = DataLoader(
                val_night_dataset, self.opt.batch_size, False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            
            self.val_day_iter = iter(self.val_day_loader)
            self.val_night_iter = iter(self.val_night_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        if not self.opt.only_im2im:
            print("There are {:d} training items and {:d} validation items\n".format(
                len(train_day_dataset) + len(train_night_dataset), len(val_day_dataset) + len(val_night_dataset)))
        else:
            print("There are {:d} training items \n".format(len(train_day_dataset) + len(train_night_dataset)))
            
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def freeze_depth(self):
        if not self.opt.only_im2im:
            for param in self.models["encoder"].parameters():
                param.requires_grad = False
            for param in self.models["depth"].parameters():
                param.requires_grad = False
    
    def unfreeze_depth(self):
        if not self.opt.only_im2im:
            for param in self.models["encoder"].parameters():
                param.requires_grad = True
            for param in self.models["depth"].parameters():
                param.requires_grad = True

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        if not self.opt.only_im2im:
            self.model_lr_scheduler.step()
        
        for scheduler in self.im2im_schedulers:
            scheduler.step()

        night_iterator = iter(self.train_night_loader)
        for batch_idx, day_inputs in enumerate(self.train_day_loader):
            try:
                night_inputs = next(night_iterator)
            except StopIteration:
                night_iterator = iter(self.train_night_loader)
                night_inputs = next(night_iterator)

            before_op_time = time.time()

            losses = self.process_batch(day_inputs, night_inputs)
            
            self.optimizer_G.zero_grad()
            self.optimizer_D.zero_grad()

            if not self.opt.only_im2im:
                self.model_optimizer.zero_grad()
                networks.freeze(self.feat_D)

            networks.freeze(self.im2im_D)
            losses["loss"].backward()
            self.optimizer_G.step()

            if not self.opt.only_im2im:
                self.model_optimizer.step()
            
            if not self.opt.only_im2im:
                networks.unfreeze(self.feat_D)
            networks.unfreeze(self.im2im_D)
            networks.freeze(self.im2im_G)
            self.freeze_depth()
            
            if not self.opt.only_im2im:
                losses["D/feat"].backward()
            
            losses["D/img"].backward()
            self.optimizer_D.step()
            if self.epoch % 5 == 0:
                if not self.opt.only_im2im:
                    for p in self.feat_D.parameters():
                        p.data.clamp_(-0.01, 0.01)

            self.unfreeze_depth()
            networks.unfreeze(self.im2im_G)

            duration = time.time() - before_op_time

            # if early_phase or late_phase:
            if batch_idx % 200 == 0:
                self.log_time(batch_idx, duration, losses)
                self.log("train", losses)
                
            self.step += 1

        if not self.opt.only_im2im:
            self.evaluate(day=True)
            self.evaluate(day=False)

    def process_batch(self, day_inputs, night_inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in day_inputs.items():
            day_inputs[key] = ipt.to(self.device)

        for key, ipt in night_inputs.items():
            night_inputs[key] = ipt.to(self.device)
        
        losses = {}
        losses["loss"] = 0
        # night2day_img = self.im2im_G(night_inputs[("color", 0, 0)])[1]
        # day2day_img = self.im2im_G(day_inputs[("color", 0, 0)])[1]

        # # Loss recons
        # losses["recons"] = self.l1loss(day_inputs[("color", 0, 0)], day2day_img)
        # losses["recons"] *= self.opt.lambda_rec_img
        # losses["loss"] += losses["recons"]

        # # Loss G
        # night2day_D = self.im2im_D(night2day_img)[0]
        # day2day_D = self.im2im_D(day2day_img)[0]
        # source_label = torch.FloatTensor(night2day_D.data.size()).fill_(0).to(self.device)
        # target_label = torch.FloatTensor(day2day_D.data.size()).fill_(1).to(self.device)
        # losses["G/img"] = self.mse(day2day_D, source_label) + self.mse(night2day_D, target_label)
        # losses["G/img"] *= self.opt.lambda_gan_img
        # losses["loss"] += losses["G/img"]
        
        # # Loss D
        # night2day_D = self.im2im_D(night2day_img.detach())[0]
        # day2day_D = self.im2im_D(day2day_img.detach())[0]
        # losses["D/img"] = self.mse(day2day_D, target_label) + self.mse(night2day_D, source_label)

        # Night to day task
        night2day_img, day2day_img, f_night, f_day, size = self.forward_G(night_inputs[("color", 0, 0)], day_inputs[("color", 0, 0)]) 
        
        # Calculate loss G
        day_real = task.scale_pyramid(day_inputs[("color", 0, 0)], size - 1)
        losses["recons"] = 0
        losses["G/img"] = 0
        for i in range(size - 1):
            losses["recons"] += self.l1loss(day2day_img[i], day_real[i])
            day_fake = self.im2im_D(night2day_img[i])
            for day_fake_i in day_fake:
                losses["G/img"] += torch.mean((day_fake_i - 1.0) ** 2)
        
        losses["recons"] *= self.opt.lambda_rec_img
        losses["G/img"] *= self.opt.lambda_gan_img
        losses["loss"] += losses["recons"]
        losses["loss"] += losses["G/img"]
        
        # Calculate loss D
        losses["D/img"] = 0
        day_fake = []
        size = len(night2day_img)
        for i in range(size):
            day_fake.append(self.fake_img_pool.query(night2day_img[i]))
        
        day_real = task.scale_pyramid(day_inputs[("color", 0, 0)], size)

        for (real_i, fake_i) in zip(day_real, day_fake):
            D_real = self.im2im_D(real_i.detach())
            D_fake = self.im2im_D(fake_i.detach())
            for (D_real_i, D_fake_i) in zip(D_real, D_fake):
                losses["D/img"] += (torch.mean((D_real_i - 1.0) ** 2) + 
                                torch.mean((D_fake_i - 0.0) ** 2)) * 0.5

        # Calculate loss depth
        if not self.opt.only_im2im:
            n2d_features = self.models["encoder"](night2day_img)
            n2d_outputs = self.models["depth"](n2d_features)

            n2d_outputs.update(self.predict_poses(night_inputs, n2d_features))
            self.generate_images_pred(night_inputs, n2d_outputs)
            losses["night"] = self.compute_losses(night_inputs, n2d_outputs)

            d2d_features = self.models["encoder"](day2day_img)
            d2d_outputs = self.models["depth"](d2d_features)

            d2d_outputs.update(self.predict_poses(day_inputs, d2d_features))
            self.generate_images_pred(day_inputs, d2d_outputs)
            losses["day"] = self.compute_losses(day_inputs, d2d_outputs)

            losses["loss"] += losses["night"] + losses["day"]
            
            for i in range(self.num_scales):
                n2d_pred = self.feat_D(n2d_features[i])[0]
                d2d_pred = self.feat_D(d2d_features[i])[0]
                source_label = torch.FloatTensor(d2d_pred.data.size()).fill_(0).to(self.device)
                target_label = torch.FloatTensor(n2d_pred.data.size()).fill_(1).to(self.device)
                losses["G/feat"] = self.mse(d2d_pred, source_label) + self.mse(n2d_pred, target_label)
                losses["G/feat"] *= self.opt.lambda_gan_img
                losses["loss"] += losses["G/feat"]

                n2d_pred = self.feat_D(n2d_features[i].detach())[0]
                d2d_pred = self.feat_D(d2d_features[i].detach())[0]
                losses["D/feat"] = self.mse(d2d_pred, target_label) + self.mse(n2d_pred, source_label)
                
        return losses
    
    def forward_G(self, img_s, img_t):

        img = torch.cat([img_s, img_t], 0)
        fake = self.im2im_G(img)

        size = len(fake)

        f_s, f_t = fake[0].chunk(2)
        img_fake = fake[1:]

        img_s_fake = []
        img_t_fake = []

        for img_fake_i in img_fake:
            img_s, img_t = img_fake_i.chunk(2)
            img_s_fake.append(img_s)
            img_t_fake.append(img_t)

        return img_s_fake, img_t_fake, f_s, f_t, size


    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            # inputs = self.val_iter.next()
            inputs = next(self.val_day_iter)
        except StopIteration:
            self.val_day_iter = iter(self.val_day_loader)
            # inputs = self.val_iter.next()
            inputs = next(self.val_day_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def evaluate(self, day=True):
        """Evaluates a pretrained model using a specified test set
        """
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        EVAL_DEPTH = 40
        self.set_eval()

        assert sum((self.opt.eval_mono, self.opt.eval_stereo)) == 1, \
            "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

        pred_disps = []
        gt = []

        print("-> Computing predictions with size {}x{}".format(
            self.opt.width, self.opt.height))

        if day:
            dataloader = self.val_day_loader
            eval_split = 'val_day'
        else:
            dataloader = self.val_night_loader
            eval_split = 'val_night'

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if self.opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = self.models["depth"](self.models["encoder"](input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if self.opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = self.batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                gt.append(np.squeeze(data['depth_gt'].cpu().numpy()))

        pred_disps = np.concatenate(pred_disps)
        if gt[-1].ndim == 2:
            gt[-1] = gt[-1][np.newaxis, :]
        gt = np.concatenate(gt)

        if self.opt.save_pred_disps:
            output_path = os.path.join(
                self.opt.load_weights_folder, "disps_{}_split.npy".format(eval_split))
            print("-> Saving predicted disparities to ", output_path)
            np.save(output_path, pred_disps)

        if self.opt.no_eval:
            print("-> Evaluation disabled. Done.")
            quit()

        # gt_path = os.path.join(self.opt.split, eval_split, "gt_depths.npz")
        # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

        print("-> Evaluating")

        if self.opt.eval_stereo:
            print("   Stereo evaluation - "
                "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
            self.opt.disable_median_scaling = True
            self.opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
        else:
            print("   Mono evaluation - using median scaling")

        errors = []
        ratios = []

        for i in range(pred_disps.shape[0]):
        # for i in range(len(pred_disps)):
            gt_depth = gt[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            if self.opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            else:
                mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= self.opt.pred_depth_scale_factor
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            
            mask2 = gt_depth <= EVAL_DEPTH
            pred_depth = pred_depth[mask2]
            gt_depth = gt_depth[mask2]
            errors.append(self.compute_errors(gt_depth, pred_depth))

        if not self.opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

        with open(self.result_path, 'a') as f:
            for i in range(len(mean_errors)):
                f.write(str(mean_errors[i])) #
                f.write('\t')
            f.write("\n")

        f.close()

        self.set_train()
        return mean_errors

    def batch_post_process_disparity(self, l_disp, r_disp):
        """Apply the disparity post-processing method as introduced in Monodepthv1
        """
        _, h, w = l_disp.shape
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
        r_mask = l_mask[:, :, ::-1]
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

    def compute_errors(self, gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | loss day: {:.5f} | loss night: {:.5f}" + \
                " | loss G/img: {:.5f} | loss D/img: {:.5f}" + \
                    " | loss G/feat: {:.5f} | loss D/feat: {:.5f}" + \
                        " | loss recons: {:.5f} |  time elapsed: {} | time left: {}"
        loss = losses["loss"].cpu().data
        loss_day = 0
        loss_night = 0
        loss_G_img = losses["G/img"].cpu().data
        loss_G_feat = 0
        loss_D_img = losses["D/img"].cpu().data
        loss_D_feat = 0
        loss_recons = losses["recons"].cpu().data

        if not self.opt.only_im2im:
            loss_day = losses["day"].cpu().data
            loss_night = losses["night"].cpu().data
            loss_G_feat = losses["G/feat"].cpu().data
            loss_D_feat = losses["D/feat"].cpu().data
        
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, \
                                loss_day, loss_night, loss_G_img, loss_D_img, \
                                loss_G_feat, loss_D_feat, loss_recons, \
                                sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if not self.opt.only_im2im:
            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                to_save = model.state_dict()
                if model_name == 'encoder':
                    # save the sizes - these are needed at prediction time
                    to_save['height'] = self.opt.height
                    to_save['width'] = self.opt.width
                    to_save['use_stereo'] = self.opt.use_stereo
                torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "{}.pth".format("feat_D"))
            torch.save(self.feat_D.state_dict(), save_path)
        
        save_path = os.path.join(save_folder, "{}.pth".format("im2im_G"))
        torch.save(self.im2im_G.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("im2im_D"))
        torch.save(self.im2im_D.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_G"))
        torch.save(self.optimizer_G.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_D"))
        torch.save(self.optimizer_D.state_dict(), save_path)


    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
    
    def load_pseudo_model(self):
        """Load model(s) from disk
        """
        self.opt.load_pseudo_model = os.path.expanduser(self.opt.load_pseudo_model)

        assert os.path.isdir(self.opt.load_pseudo_model), \
            "Cannot find folder {}".format(self.opt.load_pseudo_model)
        print("loading model from folder {}".format(self.opt.load_pseudo_model))

        for n in ["encoder", "depth"]:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_pseudo_model, "{}.pth".format(n))
            model_dict = self.pseudo_models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.pseudo_models[n].load_state_dict(model_dict)