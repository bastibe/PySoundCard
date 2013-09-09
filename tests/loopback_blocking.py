from pysoundcard import Stream

"""Loop back five seconds of audio data."""

fs = 44100
block_length = 16
s = Stream(sample_rate=fs, block_length=block_length)
s.start()
for n in range(int(fs*5/block_length)):
    s.write(s.read(block_length))
s.stop()
