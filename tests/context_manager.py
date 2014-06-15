from pysoundcard import Stream, continue_flag
import time

"""Loop back five seconds of audio data."""

def callback(in_data, out_data, time_info, status):
    out_data[:] = in_data
    return continue_flag

with Stream(sample_rate=44100, block_length=16, callback=callback):
    time.sleep(5)
