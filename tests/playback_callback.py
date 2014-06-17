import sys
import time
import numpy as np
from scipy.io.wavfile import read as wavread
from pysoundcard import Stream, continue_flag, complete_flag

"""Play an audio file."""

fs, wave = wavread(sys.argv[1])
wave = np.array(wave, dtype=np.float32)
wave /= 2 ** 15  # normalize -max_int16..max_int16 to -1..1
play_position = 0

def callback(in_data, out_data, time_info, status):
    global play_position
    out_data[:] = wave[play_position:play_position + block_length]
    # TODO: handle last (often incomplete) block
    play_position += block_length
    if play_position + block_length < len(wave):
        return continue_flag
    else:
        return complete_flag

block_length = 16
s = Stream(sample_rate=fs, block_length=block_length, callback=callback)
s.start()
while s.is_active():
    time.sleep(0.1)
