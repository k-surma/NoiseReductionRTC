import numpy as np
from aiortc.mediastreams import AudioStreamTrack


class RTCStreaming:
    class NoisyAudioTrack(AudioStreamTrack):
        def __init__(self, track, noise_level):
            super().__init__()
            self.track = track
            self.noise_level = noise_level

        async def recv(self):
            frame = await self.track.recv()
            frame.samples += np.random.normal(0, self.noise_level, frame.samples.shape)
            return frame
