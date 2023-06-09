from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        default='0000029664.png',
                        help='path to a test image or folder of images')
    parser.add_argument('--save_dir', type=str,
                        help='path to a test image or folder of images')
    parser.add_argument('--model_name', type=str,
                        default='pretrained_model/',
                        help='name of a pretrained model to use',)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--day2night",
                        help='if set, transform day to night',
                        action='store_true')
    parser.add_argument("--night2day",
                        help='if set, transform day to night',
                        action='store_true')
    parser.add_argument("--only_im2im",
                        help='if set, transform day to night',
                        action='store_true')
    parser.add_argument("--cyclegan_decoder",
                        help='if set, transform day to night',
                        action='store_true')
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # download_model_if_doesnt_exist(args.model_name)
    # model_path = os.path.join("models", args.model_name)
    model_path = args.model_name
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    day_decoder_path = os.path.join(model_path, "day_dec.pth")
    night_decoder_path = os.path.join(model_path, "night_dec.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    if not args.only_im2im:
        depth_decoder_path = os.path.join(model_path, "depth.pth")
        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(device)
        depth_decoder.eval()

    if args.cyclegan_decoder:
        day_decoder = networks.CycleGANDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))
    else:
        day_decoder = networks.LoopDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))    

    loaded_dict = torch.load(day_decoder_path, map_location=device)
    day_decoder.load_state_dict(loaded_dict)

    day_decoder.to(device)
    day_decoder.eval()

    if args.cyclegan_decoder:
        night_decoder = networks.CycleGANDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))
    else:
        night_decoder = networks.LoopDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(night_decoder_path, map_location=device)
    night_decoder.load_state_dict(loaded_dict)

    night_decoder.to(device)
    night_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            if not args.only_im2im:
                outputs = depth_decoder(features)
            else:
                outputs = None

            if args.day2night:
                results = night_decoder(features, outputs)
            
            if args.night2day:
                results = day_decoder(features, outputs)


            im = results[("render", 0)]
            im_resized = torch.nn.functional.interpolate(
                im, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]

            # Saving colormapped depth image
            disp_resized_np = im_resized.squeeze().cpu().numpy()
            # vmax = np.percentile(disp_resized_np, 95)
            # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # im = pil.fromarray(colormapped_im)

            disp_resized_np = np.moveaxis(disp_resized_np, 0, 2)
            # disp_resized_np = np.moveaxis(disp_resized_np, 1, 2)
            print(disp_resized_np.shape)
            im = pil.fromarray((disp_resized_np * 255).astype(np.uint8)).convert('RGB')
            name_dest_im = os.path.join(args.save_dir, "{}.png".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)