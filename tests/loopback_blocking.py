from pysoundcard import Stream

"""Loop back five seconds of audio data."""

fs = 44100
blocksize = 16
s = Stream(samplerate=fs, blocksize=blocksize)
s.start()
for n in range(int(fs * 5 / blocksize)):
    s.write(s.read(blocksize))
s.stop()
