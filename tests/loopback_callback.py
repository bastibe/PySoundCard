from pysoundcard import Stream, continue_flag
import time

"""Loop back five seconds of audio data."""

def callback(in_data, frame_count, time_info, status):
    return (in_data, continue_flag)

s = Stream(sample_rate=44100, block_length=16, callback=callback)
s.start()
time.sleep(5)
s.stop()
