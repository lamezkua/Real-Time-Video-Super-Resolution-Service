# Based on an example where the copyright is the following:
# Copyright 2021 Kognia Sports Intelligence
# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys

import torch
import PyNvCodec as nvc
import numpy as np

from espcn_pytorch import ESPCN

import scalene

@profile
def main(gpuID, encFilePath, dstFilePath, SCALE_FACTOR):
    gpu = torch.device(f'cuda:{gpuID}')
    dstFile = open(dstFilePath, "wb")
    nvDec = nvc.PyNvDecoder(encFilePath, gpuID)

    w = nvDec.Width()
    h = nvDec.Height()
    w_scaled = w * SCALE_FACTOR
    h_scaled = h * SCALE_FACTOR

    encoder_settings = {
        'preset': 'hq',
        'codec': 'h264',
        's': f'{w_scaled}x{h_scaled}',
        'bitrate': '10M'
    }
    nvEnc = nvc.PyNvEncoder(encoder_settings, gpuID)

    # Encoded video frame
    encoded_frame = np.ndarray(shape=(0), dtype=np.uint8)

    # PyTorch tensor the Y-component will be exported to
    Y_tensor = torch.zeros(h, w, dtype=torch.uint8,
                           device=gpu)

    upscale = nvc.PySurfaceResizer(w_scaled, h_scaled, nvc.PixelFormat.YUV420, gpuID)
    nv12_to_yuv = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, gpuID)
    yuv_to_nv12 = nvc.PySurfaceConverter(w_scaled, h_scaled, nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, gpuID)

    cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601,
                                             nvc.ColorRange.MPEG)

    model = ESPCN(scale_factor=SCALE_FACTOR).to(gpu)
    model.load_state_dict(torch.load(f'weights/espcn_{SCALE_FACTOR}x.pth', map_location=gpu))
    model.eval()

    while True:
        ## Pre-processing
        # Grab a video frame
        frame = nvDec.DecodeSingleSurface()
        if frame.Empty():
            break

        frame_yuv = nv12_to_yuv.Execute(frame, cc_ctx)

        # Place the Y component in a PyTorch tensor
        frame_yuv.PlanePtr(0).Export(Y_tensor.data_ptr(), w, gpuID)

        # Upscale using default NVENC resizer. The Y component will be
        # replaced with the superresolution version later.
        frame_scaled = upscale.Execute(frame_yuv)

        # Upscale the Y component using PyTorch
        Y_tensor_to_model = Y_tensor.view(1, 1, *Y_tensor.shape).to(torch.float32) / 255

        ## Inference
        Y_tensor_scaled = model(Y_tensor_to_model)

        ## Post-processing
        Y_tensor_scaled = Y_tensor_scaled.squeeze() * 255.0
        Y_tensor_scaled = Y_tensor_scaled.clip(0, 255).to(torch.uint8)

        # Wait for memory to become valid
        torch.cuda.synchronize(gpuID)

        # Copy the processed Y component into the VPF video frame
        frame_scaled.PlanePtr(0).Import(Y_tensor_scaled.data_ptr(), w_scaled, gpuID)

        # Encode the frame, resulting in a numpy array on the CPU
        frame_scaled_nv12 = yuv_to_nv12.Execute(frame_scaled, cc_ctx)
        success = nvEnc.EncodeSingleSurface(frame_scaled_nv12, encoded_frame)
        if success:
            # Write the frame to the output video file
            encByteArray = bytearray(encoded_frame)
            dstFile.write(encByteArray)

    # Encoder is asynchronous, so we need to flush it
    while True:
        success = nvEnc.FlushSinglePacket(encoded_frame)
        if(success):
            encByteArray = bytearray(encoded_frame)
            dstFile.write(encByteArray)
        else:
            break


if __name__ == "__main__":
    print('Real-Time Single Image and Video Super-Resolution Using '
          'an Efficient Sub-Pixel Convolutional Neural Network (ONLY GPU utilization).')
    print('Usage: test_video_onlygpu.py $input_file $output_file $scale_factor')

    if(len(sys.argv) < 4):
        print("Provide: input_file output_file scale_factor")
        exit(1)

    if torch.cuda.is_available():
        gpuID = 0
    else:
        print("There is not GPU available.")
        exit(1)
    encFilePath = sys.argv[1]
    decFilePath = sys.argv[2]
    SCALE_FACTOR = int(sys.argv[3])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    #scalene_profiler.start()

    main(gpuID, encFilePath, decFilePath, SCALE_FACTOR)

    end.record()
    print("SR took: ", start.elapsed_time(end)/1000.0)

    #scalene_profiler.stop()

    # Waits for everything to finish running
    torch.cuda.synchronize()
