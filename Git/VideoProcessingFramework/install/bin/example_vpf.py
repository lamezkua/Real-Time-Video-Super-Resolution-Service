import PyNvCodec as nvc

gpuID = 0
encFile = "big_buck_bunny_1080p_h264.mp4"
xcodeFile = open("big_buck_bunny_1080p.h264", "wb")

nvDec = nvc.PyNvDecoder(encFile, gpuID)
nvEnc = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'h264', 's': '1920x1080'}, gpuID)

while True:
    rawSurface = nvDec.DecodeSingleSurface()
    # Decoder will return zero surface if input file is over;
    if not (rawSurface.GpuMem()):
        break

    encFrame = nvEnc.EncodeSingleSurface(rawSurface)
    if(encFrame.size):
        frameByteArray = bytearray(encFrame)
        xcodeFile.write(frameByteArray)

# Encoder is asynchronous, so we need to flush it
encFrames = nvEnc.Flush()
for encFrame in encFrames:
    encByteArray = bytearray(encFrame)
    xcodeFile.write(encByteArray)
