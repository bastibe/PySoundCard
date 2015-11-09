#!/usr/bin/env python
import os
from setuptools import setup
from sys import platform
from platform import architecture

platform = os.environ.get('PYSOUNDCARD_PLATFORM', platform)
architecture = os.environ.get('PYSOUNDCARD_ARCHITECTURE', architecture()[0])

if platform == 'darwin':
    libname = 'portaudio.dylib'
elif platform == 'win32':
    libname = 'portaudio' + architecture + '.dll'
else:
    libname = None

if libname:
    packages = ['_soundcard_data']
    package_data = {'_soundcard_data': [libname, 'LICENSE']}
else:
    packages = None
    package_data = None

cmdclass ={}

try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    pass
else:
    class bdist_wheel_half_pure(bdist_wheel):
        """Create OS-dependent, but Python-independent wheels."""
        def get_tag(self):
            pythons = 'py2.py3.cp26.cp27.cp32.cp33.cp34.cp35.pp27.pp32'
            if platform == 'darwin':
                oses = 'macosx_10_5_x86_64.macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64'
            elif platform == 'win32':
                if architecture == '32bit':
                    oses = 'win32'
                else:
                    oses = 'win_amd64'
            else:
                pythons = 'py2.py3'
                oses = 'any'
            return pythons, 'none', oses

    cmdclass['bdist_wheel'] = bdist_wheel_half_pure


setup(
    name='PySoundCard',
    version='0.5.1',
    description='An audio library based on PortAudio, CFFI and NumPy',
    author='Bastian Bechtold',
    author_email='basti@bastibe.de',
    url='https://github.com/bastibe/PySoundCard',
    keywords=['audio', 'portaudio'],
    py_modules=['pysoundcard'],
    packages=packages,
    package_data=package_data,
    cmdclass=cmdclass,
    license='BSD 3-Clause License',
    install_requires=['numpy',
                      'cffi>=0.6'],
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
