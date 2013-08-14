============
PyAudio-CFFI
============

PyAudio-CFFI is an audio library based on PortAudio, CFFI and NumPy

PyAudio-CFFI can play and record audio data. Audio devices are
supported through PortAudio_, which is a free, cross-platform,
open-source audio I/O library that runs on may platforms including
Windows, OS X, and Unix (OSS/ALSA). It is accessed through CFFI_,
which is a foreign function interface for Python calling C code. CFFI
is supported for CPython 2.6+, 3.x and PyPy 2.0+. PyAudio-CFFI
represents audio data as NumPy arrays.

PyAudio-CFFI is inspired by PyAudio_. Its main difference is that it
uses CFFI instead of a CPython extension and tries to implement a more
pythonic interface. Its performance characteristics are very similar.

.. _PortAudio: http://www.portaudio.com/
.. _CFFI: http://cffi.readthedocs.org/
.. _PyAudio: http://people.csail.mit.edu/hubert/pyaudio/


| PyAudio-CFFI is BSD licensed.
| (c) 2013, Bastian Bechtold

Usage
-----

The basic building block of audio input/output in PyAudio-CFFI are
streams. Streams represent sound cards, both for audio playback and
recording. Every stream has a sample rate, a block size, an input
device and/or an output device.

A stream can be either full duplex (both input and output) or half
duplex (either input or output). This is determined by specifying one
or two devices for the stream. Both devices must be part of the same
audio API.

There are two modes of operation for streams: read/write and callback
mode.

Read/Write Mode
~~~~~~~~~~~~~~~

In read/write mode, two methods are used to play/record audio: For
playback, you ``write()`` to a stream. For recording, you ``read()``
from a stream. You can read/write up to one block of audio data to a
stream without having to wait for it to play.

Here is an example for a program that records a block of audio and
immediately plays it back:

.. include:: tests/loopback_blocking.py
   :code: python

Here is another example that reads a wave file and plays it back:

.. include:: tests/playback_blocking.py
   :code: python

Callback Mode
~~~~~~~~~~~~~

In callback mode, a callback function is defined, which will be called
asynchronously whenever there is a new block of audio data available
to read or write. The callback function must then provide/consume one
block of audio data.

Here is an equivalent example to the loopback example earlier. As you
can see, the control flow continues normally after ``s.start()`` while
the callback is running in a different thread. This is very useful for
synthesizers or filter-like audio effects.

.. include:: tests/loopback_callback.py
   :code: python

However, callback mode is somewhat burdensome for playing back audio
data from a file. Note how the callback now has to split up the audio
data into blocks and stop the stream when there is no more data
available.

.. include:: tests/playback_callback.py
   :code: python

When to use Read/Write Mode or Callback Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In general, callback mode is the more flexible and powerful way of
using PyAudio-CFFI. However, it is more complex and less performant.
Many applications will require callback mode because of its threading.
Also, it is very simple to write filter-like audio effects in callback
mode since audio input and output are readily available.

Many simple tasks, such as playing or recording a chunk of audio data
are more easily accomplished using read/write mode though. Also,
read/write runs somewhat faster and can produce/consume raw data if
requested.

If no data is read/written while in Read/Write mode, recordings are
simply discarded and silence is played. In callback mode, it is an
error not to provide audio data in the callback. Use ``numpy.zeros()``
if you want to play silence.

Performance
~~~~~~~~~~~

PyAudio-CFFI uses the CFFI library internally. Performance is a big
goal for the project. On a reasonably recent Apple computer, block
sizes of two or four samples should be no problem at a sampling rate
of 44100 or 48000 Hz.

However, performance is strongly influenced by the API in use. Also,
some combinations of audio devices can be problematic even if they are
part of the same API. In general, try to open full duplex streams only
on input/output devices of the same physical sound card for maximum
performance.
