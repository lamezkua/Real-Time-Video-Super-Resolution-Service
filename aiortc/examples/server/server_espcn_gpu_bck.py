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
#import torchvision.transforms as transforms
from espcn_pytorch import ESPCN
import torch

model = None

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

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
            global model

            # create model
            model = ESPCN(scale_factor=sf).to(device)

            # Load state dicts
            model.load_state_dict(torch.load(weight, map_location=device))

            # Set model eval mode
            model.eval()

            self.transform_selected = self.transform

        #@profile
        def SuperResolution(sf):
            global model

            # frame is in YUV format.
            # Y = luminance: black & white TV (or Luma, or gamma)
            # U = color information
            # V = color information

            data = frame.to_ndarray()

            # y = 480 x 640
            y = data[:frame.height]

            # uv = 2, 230, 320
            uv = data[frame.height:].reshape(2, frame.height // 2, frame.width // 2)

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
            new_data = np.vstack((y_model, uv.reshape(-1, sf * frame.width)))

            # Superres only the Y channel (expensive).
            # Bicubic scaling on U and V channels (cheap-ish)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(new_data, format="yuv420p")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            return new_frame

        if self.transform == "twotimes":
            sf = 2
            weight = 'weights/espcn_2x.pth'
            #print("---------------------TWO times selected------------------------")
        elif self.transform == "threetimes":
            sf = 3
            weight = 'weights/espcn_3x.pth'
            #print("---------------------THREE times selected------------------------")
        elif self.transform == "fourtimes":
            sf = 4
            weight = 'weights/espcn_4x.pth'
            #print("---------------------FOUR times selected------------------------")
        elif self.transform == "eighttimes":
            sf = 8
            weight = 'weights/espcn_8x.pth'
            #print("---------------------EIGHT times selected------------------------")
        else:
            return frame

        #print("---------------> self.transform, self.transform_selected: ", self.transform, self.transform_selected)
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
