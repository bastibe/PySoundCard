import sys
import numpy as np
from scipy.io.wavfile import read as wavread
from pysoundcard import Stream

"""Play an audio file."""

fs, wave = wavread(sys.argv[1])
wave = np.array(wave, dtype=np.float32)
wave /= 2 ** 15  # normalize -max_int16..max_int16 to -1..1

blocksize = 16
s = Stream(samplerate=fs, blocksize=blocksize)
s.start()
s.write(wave)
s.stop()
