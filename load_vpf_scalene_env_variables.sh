#!/bin/bash

PATH=/home/gpuadmin/anaconda3/envs/vpf2_scalene/bin:/home/gpuadmin/Git/FFmpeg/build_x64_release_shared/bin:$PATH

PATH_TO_FFMPEG=~/Git/FFmpeg/build_x64_release_shared
cd ~/Git/VideoProcessingFramework/install/bin
export LD_LIBRARY_PATH=$PATH_TO_FFMPEG/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

