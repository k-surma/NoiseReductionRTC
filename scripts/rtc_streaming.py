from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from pygments.lexers import web

from scripts.data_preparation import DataPreparation


class NoisyAudioTrack(MediaStreamTrack):
    """A MediaStreamTrack that applies noise to an audio stream."""
    kind = "audio"

    def __init__(self, source, noise_level=0.1):
        super().__init__()
        self.source = source
        self.noise_level = noise_level

    async def recv(self):
        frame = await self.source.recv()
        prbs_noise = DataPreparation.generate_prbs(len(frame.samples)) * self.noise_level
        frame.samples += prbs_noise.astype(frame.samples.dtype)
        return frame

async def offer(request):
    pc = RTCPeerConnection()

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            noisy_track = NoisyAudioTrack(track, noise_level=0.1)
            pc.addTrack(noisy_track)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
