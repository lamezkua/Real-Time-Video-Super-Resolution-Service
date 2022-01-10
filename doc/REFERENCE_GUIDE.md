## Reference Guide

The code in this repository can do the video super-resolution from two different inputs: I. a video-file, and II. real-time.

I. When having a video-file as the input, one of the following three command sets is to be followed to do the super-resolution:

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


II. When having real-time input, one of the following two command sets is to be followed to do the super-resolution:

1. CPU and GPU code (the default aiortc codec is used)

On the server side,
``` shell
conda env create -f aiortc2_environment.yml
conda activate aiortc2
cd ./aiortc/examples/server
# a digital certificate should be configured
python server_espcn_gpu.py --cert-file=fullchain.pem --key-file=privkey.pem
```

On the client side,
a. Open a browser window and provide the following Uniform Resource Locator (URL) and port to start the client side 
   (e.g., https://video-sr.cs.aalto.fi:8080/).
b. Select from the displayed user interface (UI) the desired parameters to start the service.

2. GPU code (the default aiortc codec is used)

On the server side,
``` shell
conda env create -f vpf2_aiortc2_environment.yml
conda activate vpf2_aiortc2
~ ./load_vpf_env_variables.sh
# a digital certificate should be configured
python server_espcn_gpu_vpf.py --cert-file=fullchain.pem --key-file=privkey.pem
```

On the client side,
a. Open a browser window and provide the following Uniform Resource Locator (URL) and port to start the client side 
   (e.g., https://video-sr.cs.aalto.fi:8080/).
b. Select from the displayed user interface (UI) the desired parameters to start the service.
