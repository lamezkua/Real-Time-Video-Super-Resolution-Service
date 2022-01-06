import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

#SR specific
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from espcn_pytorch import ESPCN
import torch

import PyNvCodec as nvc

model = None

# setting device on GPU if available
if torch.cuda.is_available():
    gpuID = 0
else:
    print("There is not GPU available.")
    exit(1)

gpu = torch.device(f'cuda:{gpuID}')

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"
    transform_selected = None

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()

        # Consume all available frames.
        # If we don't do this, we'll bloat indefinitely.
        while not self.track._queue.empty():
            frame = await self.track.recv()

        def SuperResolution_init(sf, weight):
            global model, w, h, w_scaled, h_scaled, nvEnc, encoded_frame, Y_tensor, upscale, nv12_to_yuv, yuv_to_nv12, cc_ctx

            w = frame.width
            h = frame.height
            w_scaled = w * sf
            h_scaled = h * sf

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
            Y_tensor = torch.zeros(h, w, dtype=torch.uint8, device=gpu)

            upscale = nvc.PySurfaceResizer(w_scaled, h_scaled, nvc.PixelFormat.YUV420, gpuID)
            nv12_to_yuv = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, gpuID)
            yuv_to_nv12 = nvc.PySurfaceConverter(w_scaled, h_scaled, nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, gpuID)

            cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

            model = ESPCN(scale_factor=sf).to(gpu)
            model.load_state_dict(torch.load(weight, map_location=gpu))
            model.eval()


            self.transform_selected = self.transform

        def SuperResolution(sf):
            global model

            data = frame.to_ndarray()

            # Load frame in numpyarray format to GPU
            nvUpl_yuv = nvc.PyFrameUploader(int(w), int(h), nvc.PixelFormat.YUV420, gpuID)
            frame_yuv = nvUpl_yuv.UploadSingleFrame(data)

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

            # Download frame in Surface format from GPU
            nvDwn = nvc.PySurfaceDownloader(w_scaled, h_scaled, nvc.PixelFormat.YUV420, gpuID)
            new_data = np.ndarray(shape=(h_scaled + int(h_scaled / 2), w_scaled), dtype=np.uint8)
            nvDwn.DownloadSingleSurface(frame_scaled, new_data)

            # Rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(new_data, format="yuv420p")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            return new_frame

        if self.transform == "twotimes":
            sf = 2
            weight = 'weights/espcn_2x.pth'
        elif self.transform == "threetimes":
            sf = 3
            weight = 'weights/espcn_3x.pth'
        elif self.transform == "fourtimes":
            sf = 4
            weight = 'weights/espcn_4x.pth'
        elif self.transform == "eighttimes":
            sf = 8
            weight = 'weights/espcn_8x.pth'
        else:
            return frame

        if model is None or self.transform != self.transform_selected:
            print(f'Doing superresolution with scaling factor of {sf}x')
            SuperResolution_init(sf, weight)

        return SuperResolution(sf)


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.write_audio:
        recorder = MediaRecorder(args.write_audio)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            local_video = VideoTransformTrack(
                track, transform=params["video_transform"]
            )
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )

    torch.cuda.synchronize()
