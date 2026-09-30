"""K-major weight blob helpers shared by the golden generators.

The 64-lane row reads weights K-major so one K index's lane weights arrive in
a single SRAM beat; the systolic array still reads OC-major. This mirrors
conv_weights_to_blob / dw_weights_to_blob in tools/model_packer.py, but works
on the flat arrays the generators already build.
"""

import numpy as np

OC_MAJOR = 0
K_MAJOR = 1
MAC_LANES = 64


def conv_blob(weight_ohwi, out_c, k_depth, layout=K_MAJOR):
    """[out_c][kh][kw][in_c] -> the on-chip blob, same dtype and byte count."""
    flat = np.asarray(weight_ohwi).reshape(out_c, k_depth)
    if layout != K_MAJOR:
        return flat.reshape(-1)
    parts = [flat[g:g + MAC_LANES].T.reshape(-1)
             for g in range(0, out_c, MAC_LANES)]
    return np.concatenate(parts) if parts else flat.reshape(-1)


def dw_blob(weight_chw, n_ch, layout=K_MAJOR):
    """[ch][kh][kw] -> tap-major [kh*kw][ch] for the row, unchanged otherwise."""
    flat = np.asarray(weight_chw).reshape(n_ch, -1)
    if layout != K_MAJOR:
        return flat.reshape(-1)
    return flat.T.reshape(-1)


def words_from_bytes(byte_arr):
    """Pad to a word boundary and split into little-endian uint32 words."""
    byte_arr = bytes(byte_arr)
    byte_arr += b'\x00' * ((4 - len(byte_arr) % 4) % 4)
    return [int.from_bytes(byte_arr[i:i + 4], 'little')
            for i in range(0, len(byte_arr), 4)]
