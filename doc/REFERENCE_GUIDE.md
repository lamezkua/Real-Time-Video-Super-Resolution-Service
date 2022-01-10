## Reference Guide

The code in this repository can do the video super-resolution from two different inputs: I. a video-file, and II. real-time.

I. When having a video-file as the input, follow one of the following three command sets to do the super-resolution:

1. Software Codec with CPU and GPU code

``` shell
conda env create -f espcn_cpu_environment.yml
conda activate espcn_cpu
cd ./ESPCN-PyTorch
# comment and un-comment the corresponding codec lines
python test_video_cpu5.py --file ./data/file_example_MP4_480_1_5MG.mp4 --scale-factor 2 --weights ./weights/espcn_2x.pth --cuda
```

2. Hardware Codec with CPU and GPU code

``` shell
conda env create -f espcn_gpu_environment.yml
conda activate espcn_gpu
cd ./ESPCN-PyTorch
# comment and un-comment the corresponding codec lines
python test_video_cpu5.py --file ./data/file_example_MP4_480_1_5MG.mp4 --scale-factor 2 --weights ./weights/espcn_2x.pth --cuda
```

3. Hardware Codec with GPU code

``` shell
conda env create -f vpf2_environment.yml
conda activate vpf2
~ ./load_vpf_env_variables.sh
python test_video_onlygpu.py data/file_example_MP4_480_1_5MG.mp4 file_example_MP4_480_1_5MG_out.h264 2
```

