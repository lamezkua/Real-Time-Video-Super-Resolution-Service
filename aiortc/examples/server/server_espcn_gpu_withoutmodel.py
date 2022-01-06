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
import torchvision.transforms as transforms
from PIL import Image
from espcn_pytorch import ESPCN

model = None
pil2tensor = None
tensor2pil =  None
device = torch.device("cuda:0")

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()

        def Upscale(sf):
            #SR
            img1 = frame.to_ndarray(format="bgr24")

            new_size = (426 * sf, 240 * sf)

            # resize image
            img2 = cv2.resize(img1, new_size, interpolation = cv2.INTER_CUBIC)

            # rebuild a VideoFrame, preserving timing information
            upscaled_frame = VideoFrame.from_ndarray(img2, format="bgr24")
            upscaled_frame.pts = frame.pts
            upscaled_frame.time_base = frame.time_base
            return upscaled_frame

        def SuperResolution_init(sf, weight):
            global model
            global pil2tensor

            # create model
            model = ESPCN(scale_factor=sf).to(device)

            # Load state dicts
            model.load_state_dict(torch.load(weight, map_location=device))

            # Set model eval mode
            model.eval()

            # img preprocessing operation
            pil2tensor = transforms.ToTensor()
            tensor2pil = transforms.ToPILImage()

        def SuperResolution():
            global model
            global pil2tensor

            #SR
            img1 = frame.to_ndarray(format="bgr24")

            new_size = (426, 240)

            # resize image
            img2 = cv2.resize(img1, new_size, interpolation = cv2.INTER_CUBIC)

            img = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).convert("YCbCr")
            y, cb, cr = img.split()
            img = pil2tensor(y).view(1, -1, y.size[1], y.size[0])
            img = img.to(device)

            with torch.no_grad():
                prediction = model(img)

            prediction = prediction.cpu()
            sr_frame_y = prediction[0].detach().numpy()
            sr_frame_y *= 255.0
            sr_frame_y = sr_frame_y.clip(0, 255)
            sr_frame_y = Image.fromarray(np.uint8(sr_frame_y[0]), mode="L")

            sr_frame_cb = cb.resize(sr_frame_y.size, Image.BICUBIC)
            sr_frame_cr = cr.resize(sr_frame_y.size, Image.BICUBIC)
            sr_frame = Image.merge("YCbCr", [sr_frame_y, sr_frame_cb, sr_frame_cr]).convert("RGB")
            # before converting the result in RGB
            sr_frame = cv2.cvtColor(np.asarray(sr_frame), cv2.COLOR_RGB2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(sr_frame, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        if self.transform == "twotimes":
            sf = 2
            weight = 'weights/espcn_2x.pth'
            print("---------------------TWO times selected------------------------")
        elif self.transform == "threetimes":
            sf = 3
            weight = 'weights/espcn_3x.pth'
            print("---------------------THREE times selected------------------------")
        elif self.transform == "fourtimes":
            sf = 4
            weight = 'weights/espcn_4x.pth'
            print("---------------------FOUR times selected------------------------")
        elif self.transform == "eighttimes":
            sf = 8
            weight = 'weights/espcn_8x.pth'
            print("---------------------EIGHT times selected------------------------")
        else:
            return frame

        if model is None:
            SuperResolution_init(sf, weight)

        #return SuperResolution()
        return Upscale(sf)


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
