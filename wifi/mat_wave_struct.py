import scipy.io
import numpy as np

from prelude import CNDarray

class WaveStruct:
    def __init__(self, path: str) -> None:
        wifi_mat = scipy.io.loadmat(path)["waveStruct"]
        self.type: str = wifi_mat["type"][0,0][0]
        self.fs: float = wifi_mat["Fs"][0, 0][0][0]
        self.waveform: CNDarray = wifi_mat["waveform"][0, 0].flatten()

        # these fields remain to be used in later edition
        self.config = wifi_mat["config"][0, 0]
        self.impairments = wifi_mat["impairments"][0, 0]
