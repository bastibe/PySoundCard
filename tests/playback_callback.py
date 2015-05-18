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

blocksize = 16
s = Stream(samplerate=fs, blocksize=blocksize, callback=callback)
s.start()
while s.is_active():
    time.sleep(0.1)
