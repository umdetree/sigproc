"""
use square root raised cosine pulse shaping to generate multiband signals
"""

import numpy as np
import h5py

from ..filters import SqrtRaisedCosFilter
from ..cs import multicoset

def main():
    qpsk_symbols = np.exp(1j * np.pi * np.array([1, 3, 7, 5]) / 4)
    data_len = 1000
    n_symbs = 8
    n_sigs = 6
    n_bands = 16
    bw = 50e6
    n_bands_tot = 40
    # multicoset sampling channels
    n_chs = 40

    sps = int(round(bw * n_bands_tot / 20e6))
    filter = SqrtRaisedCosFilter(
        fsymb=20e6,
        rolloff=0.35,
        sps=sps,
    )

    symbols = np.random.randint(0, len(qpsk_symbols), (data_len, n_symbs))
    symbols = qpsk_symbols[symbols]

    sig_len = n_symbs * sps
    signals = np.zeros((data_len, n_chs, int(sig_len / n_chs)), dtype=np.cdouble)
    occupied_bands = np.zeros((data_len, n_bands), dtype=np.uint8)
    for i in range(data_len):
        bids = np.random.choice(n_bands, n_sigs, replace=False)
        occupied_bands[i][bids] = 1
        signal = np.zeros(sig_len, dtype=np.cdouble)
        for bid in bids:
            fc = (bid + 0.5) * bw
            signal += filter.pulse_shape(symbols[i], fc)

        # multicoset sampling
        sub_samples, _ = multicoset.sample(
            signal,
            0,
            0,
            n_bands_tot,
            n_chs,
            offsets=multicoset.DEFAULT_OFFSETS,
        )
        signals[i] = sub_samples

    # then save signals and occupied_bands
    # with h5py.File("multiband.h5", "r") as file:
    #     file.create_dataset("waveform", (data_len, sig_len), dtype=np.cdouble)
    #     file.create_dataset("label", (data_len, n_bands), dtype=np.uint8)


if __name__ == "__main__":
    main()
