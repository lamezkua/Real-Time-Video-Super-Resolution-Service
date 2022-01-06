# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.nn.functional import interpolate
from PIL import Image
from tqdm import tqdm

from espcn_pytorch import ESPCN

parser = argparse.ArgumentParser(description="Real-Time Single Image and Video Super-Resolution Using "
                                             "an Efficient Sub-Pixel Convolutional Neural Network.")
parser.add_argument("--file", type=str, required=True,
                    help="Test low resolution video name.")
parser.add_argument("--weights", type=str, required=True,
                    help="Generator model name. ")
parser.add_argument("--scale-factor", type=int, required=True, choices=[2, 3, 4, 8],
                    help="Super resolution upscale factor. (default:4)")
parser.add_argument("--view", action="store_true",
                    help="Super resolution real time to show.")
parser.add_argument("--cuda", action="store_true",
                    help="Enables cuda")

args = parser.parse_args()
print(args)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = ESPCN(scale_factor=args.scale_factor).to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Set model eval mode
model.eval()

# img preprocessing operation
pil2tensor = transforms.ToTensor()
#tensor2pil = transforms.ToPILImage()

# Open video file
video_name = args.file
print(f"Reading `{os.path.basename(video_name)}`...")
video_capture = cv2.VideoCapture(video_name)
# Prepare to write the processed image into the video.
fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
# Set video size
size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sr_size = (size[0] * args.scale_factor, size[1] * args.scale_factor)
#pare_size = (sr_size[0] * 2 + 10, sr_size[1] + 10 + sr_size[0] // 5 - 9)
# Video write loader.
srgan_writer = cv2.VideoWriter(f"espcn_{args.scale_factor}x_{os.path.basename(video_name)}",
                               cv2.VideoWriter_fourcc(*"MPEG"), fps, sr_size)
#compare_writer = cv2.VideoWriter(f"compare_{args.scale_factor}x_{os.path.basename(video_name)}",
#                                 cv2.VideoWriter_fourcc(*"MPEG"), fps, pare_size)

# read frame
success, raw_frame = video_capture.read()
progress_bar = tqdm(range(total_frames), desc="[processing video and saving/view result videos]")

e1 = cv2.getTickCount()

for index in progress_bar:
    if success:

        # Pre-processing
        #img = raw_frame.to_image().convert("YCbCr")
        #print("Raw Frame shape: -------------->",raw_frame.shape)
        #print("Raw Frame size: -------------->",raw_frame.size)
        img = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)).convert("YCbCr")
        y, cb, cr = img.split()
        img = pil2tensor(y).view(1, -1, y.size[1], y.size[0])
        img = img.to(device)

        # Inference
        with torch.no_grad():
            prediction = model(img)
        # Post-processing

        sr_frame_y = prediction[0].detach()
        sr_frame_y *= 255.0
        #sr_frame_y = sr_frame_y.clip(0, 255)
        sr_frame_y = torch.clamp(sr_frame_y, min=0, max=255)
        sr_frame_y = Image.fromarray(sr_frame_y[0].byte().cpu().numpy(), mode="L")

        cb_img = pil2tensor(cb).view(1, -1, cb.size[1], cb.size[0]).to(device)
        cr_img = pil2tensor(cr).view(1, -1, cr.size[1], cr.size[0]).to(device)
        cb_img = interpolate(cb_img, scale_factor=args.scale_factor, mode='bicubic')
        cr_img = interpolate(cr_img, scale_factor=args.scale_factor, mode='bicubic')
        cb_img *= 255.0
        cr_img *= 255.0
        #cb_img = cb_img.clip(0, 255)
        cb_img = torch.clamp(cb_img, min=0, max=255)
        #cr_img = cr_img.clip(0, 255)
        cr_img = torch.clamp(cr_img, min=0, max=255)
        sr_frame_cb = Image.fromarray(cb_img[0][0].byte().cpu().numpy(), mode="L")
        sr_frame_cr = Image.fromarray(cr_img[0][0].byte().cpu().numpy(), mode="L")
        sr_frame = Image.merge("YCbCr", [sr_frame_y, sr_frame_cb, sr_frame_cr]).convert("RGB")

        # before converting the result in RGB
        sr_frame = cv2.cvtColor(np.asarray(sr_frame), cv2.COLOR_RGB2BGR)
        # save sr video
        srgan_writer.write(sr_frame)

        if args.view:
            # display video
            cv2.imshow("LR video convert HR video ", final_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # next frame
        success, raw_frame = video_capture.read()

e2 = cv2.getTickCount()
t = (e2 - e1) / cv2.getTickFrequency()
print("SR took: " +  str(t) )
