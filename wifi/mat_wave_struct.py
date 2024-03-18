import scipy.io
import numpy as np

from prelude import CNDarray

def autocorrelate(waveform: CNDarray, ws: int, n_st: int, n: int):
    rs = np.sum(waveform[n: n + ws] * waveform[n + n_st: n + n_st + ws].conj())
    d = np.sum(waveform[n: n + ws] * waveform[n: n + ws].conj())
    return rs / d

class WaveStruct:
    def __init__(self, path: str) -> None:
        wifi_mat = scipy.io.loadmat(path)["waveStruct"]
        self.type: str = wifi_mat["type"][0,0][0]
        self.fs: float = wifi_mat["Fs"][0, 0][0][0]
        self.n_20mhz = int(round(self.fs / 20e6))
        self.waveform: CNDarray = wifi_mat["waveform"][0, 0].flatten()

        # these fields remain to be used in later edition
        self.config = wifi_mat["config"][0, 0]
        self.impairments = wifi_mat["impairments"][0, 0]

    def find_start(self) -> np.intp:
        n_st = int(round(0.8e-6 * self.fs))
        rs = [autocorrelate(self.waveform, 9 * n_st, n_st, n) for n in range(10 * n_st)]
        id = np.argmax(np.abs(rs))
        return id

    def lsig_raw_bits(self, start_id: int):
        lsig_start = int(round(16.8e-6 * self.fs))
        lsig_end = int(round(20e-6 * self.fs))
        lsig_waveform = self.waveform[start_id + np.arange(lsig_start, lsig_end)]
        lsig_f = np.fft.fft(lsig_waveform)

        symbs_start = -32 * self.n_20mhz
        symbs_end = symbs_start + 64
        symbs = lsig_f[symbs_start : symbs_end] if symbs_end < 0 else np.concatenate((lsig_f[symbs_start:], lsig_f[:symbs_end]))

        symbs = np.roll(symbs, 32)
        # remove pilot
        symbs = np.concatenate((
            symbs[-26:-21],
            symbs[-20:-7],
            symbs[-6:],
            symbs[1:7],
            symbs[8:21],
            symbs[22:27],
        ))

        return np.array(symbs.real > 0, dtype=np.uint8)
