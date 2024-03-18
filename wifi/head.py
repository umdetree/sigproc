import numpy as np
import matplotlib.pyplot as plt

from wifi.mat_wave_struct import WaveStruct
import wifi.viterbi as viterbi


def indices_of_symbol(n: int):
    return np.arange(64) + 400 + n * 80 + 16


def interleave(raw_bits: np.ndarray, n_bpsc: int):
    # carried bits per symbol
    n_cbps = len(raw_bits)
    s = max(n_bpsc // 2, 1)
    k_ids = np.arange(n_cbps)
    i_ids = int(n_cbps / 16) * (k_ids % 16) + k_ids // 16
    j_ids = s * (i_ids // s) + (i_ids + n_cbps - 16 * i_ids // n_cbps) % s
    res = np.zeros_like(raw_bits, dtype=int)
    res[j_ids] = raw_bits
    return res


def deinterleave(raw_bits: np.ndarray, n_bpsc: int):
    # carried bits per symbol
    n_cbps = len(raw_bits)
    s = max(n_bpsc // 2, 1)
    j_ids = np.arange(n_cbps)
    i_ids = s * (j_ids // s) + (j_ids + 16 * j_ids // n_cbps) % s
    k_ids = 16 * i_ids - (n_cbps - 1) * (16 * i_ids // n_cbps)
    res = np.zeros_like(raw_bits, dtype=int)
    res[k_ids] = raw_bits
    return res


def parse_signal_bits(bits: np.ndarray):
    parity = bits[:17].sum() % 2
    if parity != bits[17]:
        raise ValueError("signal parity check failed")

    rate = bits[:4]
    print("RATE:", rate)
    length_bits = np.concatenate((bits[5:17], np.zeros(4, dtype=np.uint8)))
    byte_array = np.packbits(length_bits[::-1])
    length = int.from_bytes(byte_array, "big")
    print("LENGTH:", length)


def test_interleave_signal():
    wave_struct = WaveStruct("./wifi/normal.mat")
    raw_bits = wave_struct.lsig_raw_bits(0)
    raw_bits = np.tile(raw_bits, 1)
    print(raw_bits)
    deinterleaved_bits = deinterleave(raw_bits, n_bpsc=1)
    print(deinterleaved_bits)
    reinterleaved_bits = interleave(deinterleaved_bits, n_bpsc=1)
    assert np.all(reinterleaved_bits == raw_bits)
    print("test interleave pass")


def test_decode_signal():
    # file = "./wifi/mcs3_501.mat"
    file = "./wifi/40ht_mcs2_1024.mat"
    print(file)
    wave_struct = WaveStruct(file)
    raw_bits = wave_struct.lsig_raw_bits(0)
    deinterleaved_bits = deinterleave(raw_bits, n_bpsc=1)
    decoded_bits = viterbi.decode(deinterleaved_bits)
    print(decoded_bits)
    parse_signal_bits(decoded_bits)


def main():
    wave_struct = WaveStruct("./wifi/mcs3_1024.mat")
    waveform = wave_struct.waveform * 1.0
    start_id = wave_struct.find_start()
    print(f"start id: {start_id}")
    # matlab is good
    assert start_id == 0

    waveform = waveform[start_id:]
    signal_start = 320 + 16
    signal_end = 400
    print(f"signal start: {signal_start}")
    print(f"signal end: {signal_end}")
    signal = waveform[2 * signal_start : 2 * signal_end]
    signal_f = np.fft.fft(signal)
    signal_f = np.concatenate((signal_f[-26:], signal_f[1:26]))
    plt.plot(signal_f.real)
    plt.plot(signal_f.imag)
    plt.show()
    plt.plot(signal_f.real, signal_f.imag, ".")
    plt.show()

    for n in range(20):
        symbol = waveform[indices_of_symbol(n)] * 1.0
        symbol = np.fft.fft(symbol)
        symbol = np.concatenate(
            (
                symbol[-26:-21],
                symbol[-20:-7],
                symbol[-6:],
                symbol[1:7],
                symbol[8:21],
                symbol[22:26],
            )
        )
        plt.plot(symbol.real, symbol.imag, "b.")
    plt.show()


if __name__ == "__main__":
    test_decode_signal()
    test_interleave_signal()
