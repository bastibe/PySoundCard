from pysoundcard import Stream, continue_flag
import time

"""Loop back five seconds of audio data."""

def callback(in_data, out_data, time_info, status):
    out_data[:] = in_data
    return continue_flag

s = Stream(samplerate=44100, blocksize=16, callback=callback)
s.start()
time.sleep(5)
s.stop()
