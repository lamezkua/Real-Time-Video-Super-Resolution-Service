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

import subprocess as sp

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
# Video write loader.
srgan_writer = cv2.VideoWriter(f"espcn_{args.scale_factor}x_{os.path.basename(video_name)}",
                               cv2.VideoWriter_fourcc(*"MPEG"), fps, sr_size)

# read frame
##success, raw_frame = video_capture.read()
success = False
progress_bar = tqdm(range(total_frames), desc="[processing video and saving/view result videos]")

# video properties
resolution = (size[1], size[0])
framesize = resolution[0]*resolution[1]*3

###FFMPEG_BIN = "ffmpeg"
FFMPEG_BIN = sp.getoutput("which ffmpeg")

# set up reading pipe
command = [FFMPEG_BIN,
#           '-vcodec', 'h264_cuvid',
           '-vcodec', 'h264',
           '-i', video_name,
           '-f', 'image2pipe',
           '-pix_fmt', 'bgr24',
           '-vcodec', 'rawvideo', '-an', '-']

pipe_r = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

size_frame_string = str(size[0] * args.scale_factor) + 'x' + str(size[1] * args.scale_factor)
#print("size_frame_string:", size_frame_string)
output_video_file = f"espcn_{args.scale_factor}x_{os.path.basename(video_name)}"
#print("output_video_file: ", output_video_file)
fps_string = str(int(fps))
#print("fps_string: ", fps_string)
# set up writing pipe
command = [FFMPEG_BIN,
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', size_frame_string, # size of one frame
        '-pix_fmt', 'bgr24',
#        '-r', fps_string, # frames per second
        '-r', '24', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-f', 'rawvideo',
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'libx264',
#        '-vcodec', 'h264_nvenc',
#        output_video_file]
         'test.h264']
#        'test.mkv']

pipe_w = sp.Popen(command, stdin=sp.PIPE)

# read first frame
raw_frame = pipe_r.stdout.read(framesize)

if len(raw_frame) != (size[0]*size[1]*3):
    print('Error reading frame!!!')  # Break the loop in case of an error (too few bytes were read).
    exit()
else:
    success = True

e1 = cv2.getTickCount()

for index in progress_bar:
    if success:

        # Pre-processing
        frame = np.fromstring(raw_frame, np.uint8)
        frame = frame.reshape((size[1], size[0], 3))
        #print("Frame size: -------------->",frame.size)
        #print("Frame shape: -------------->",frame.shape)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("YCbCr")
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
        #print("sr_frame size: -------------->",sr_frame.size)
        #print("sr_frame shape: -------------->",sr_frame.shape)
        #print("sr_frame type: -------------->",type(sr_frame))
        # save sr video
        #srgan_writer.write(sr_frame)
        pipe_w.stdin.write(sr_frame.tostring())

        if args.view:
            # display video
            cv2.imshow("LR video convert HR video ", final_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        pipe_r.stdout.flush()
        pipe_w.stdin.flush()
        torch.cuda.synchronize(device)

        # next frame
        ##success, raw_frame = video_capture.read()
        raw_frame = pipe_r.stdout.read(framesize)

        if len(raw_frame) != (size[0]*size[1]*3):
            print('Error reading frame!!!')  # Break the loop in case of an error (too few$
            #exit()
        else:
            success = True

e2 = cv2.getTickCount()
t = (e2 - e1) / cv2.getTickFrequency()
print("SR took: " +  str(t) )

# Wait one more second and terminate the sub-process
try:
    pipe_r.wait(1)
except (sp.TimeoutExpired):
    pipe_r.terminate()

try:
    pipe_w.wait(1)
except (sp.TimeoutExpired):
    pipe_w.terminate()

cv2.destroyAllWindows()
