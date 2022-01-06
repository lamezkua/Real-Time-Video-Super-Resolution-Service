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
#import torchvision.transforms as transforms
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
#pil2tensor = transforms.ToTensor()
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
for index in progress_bar:
    if success:
        print("raw_frame shape: ", raw_frame.shape)
        print("raw_frame type: ",type(raw_frame))
        img0 = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)).convert("YCbCr")
        print("img0 size: ", img0.size)
        print("img0 type: ",type(img0))
        img = np.array(img0)
        img = img[:, :, 0]
        print("img shape: ", img.shape)
        print("img type: ",type(img))
        #img = raw_frame.convert("YCbCr")

        # y = 480 x 640
        y = img[:size[1]]

        # uv = 2, 230, 320
        uv = img[size[1]:].reshape(2, size[1] // 2, size[0] // 2)

        # The input dimensions are interpreted in the form:
        # mini-batch x channels x [optional depth] x [optional height] x width.
        #
        # y = 480 x 640
        y = y.reshape(1, 1, *y.shape)
        y = torch.tensor(y, dtype=torch.float32).to(device) / 255  # convert to PyTorch Tensor
        #y = torch.nn.functional.interpolate(y, scale_factor=2, mode='bicubic', align_corners=False)
        with torch.no_grad():
            y_model = model(y)
        #y_model = y_model.to(torch.uint8).cpu().numpy().squeeze()  # convert back to NumPy array
        #y_model = y_model.squeeze() * 255.0
        y_model = torch.squeeze(y_model) * 255.0
        #print("-------------------------> y_model shape: ",y_model.shape)
        # y_model = y_model.clip(0, 255).to(torch.uint8).cpu().numpy()
        y_model = torch.clamp(y_model, min=0, max=255).to(torch.uint8).cpu().numpy()

        # Wait for memory to become valid
        torch.cuda.synchronize(device)

        # uv = 2 x 240 x 320
        uv = uv.reshape(1, *uv.shape)
        uv = torch.tensor(uv, dtype=torch.float32).to(device)  # convert to PyTorch Tensor
        uv = torch.nn.functional.interpolate(uv, scale_factor=sf, mode='bicubic', align_corners=False)
        uv = uv.to(torch.uint8).cpu().numpy().squeeze()  # convert back to NumPy array

        # Merge Y and UV
        new_img = np.vstack((y_model, uv.reshape(-1, sf * size[0])))


        # before converting the result in RGB
        sr_frame = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)
        # save sr video
        srgan_writer.write(sr_frame)

        if args.view:
            # display video
            cv2.imshow("LR video convert HR video ", final_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # next frame
        success, raw_frame = video_capture.read()
