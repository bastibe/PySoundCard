#!/usr/bin/env python

from distutils.core import setup

setup(
    name='PySoundCard',
    version='0.3',
    description='An audio library based on PortAudio, CFFI and NumPy',
    author='Bastian Bechtold',
    author_email='basti@bastibe.de',
    url='https://github.com/bastibe/PySoundCard',
    keywords=['audio', 'portaudio'],
    py_modules=['pysoundcard'],
    license='BSD 3-Clause License',
    requires=['numpy',
              'cffi (>=0.6)'],
    platforms='any',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Other Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Multimedia :: Sound/Audio'
    ],
    long_description='''
    PySoundCard can play and record audio.

    Audio devices are supported through PortAudio_, which is a free,
    cross-platform, open-source audio I/O library that runs on many
    operating systems including Windows, OS X and Linux. It is
    accessed through CFFI_, which is a foreign function interface for
    Python calling C code. CFFI is supported for CPython 2.6+, 3.x and
    PyPy 2.0+. PySoundCard represents audio data as NumPy arrays.

    You must have PortAudio installed in order to run PySoundCard.

    .. _PortAudio: http://www.portaudio.com/
    .. _CFFI: http://cffi.readthedocs.org/
    ''')
