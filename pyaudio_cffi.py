from cffi import FFI
import atexit
import numpy as np

ffi = FFI()
ffi.cdef("""
typedef int PaError;
PaError Pa_Initialize(void);
PaError Pa_Terminate(void);
int Pa_GetVersion(void);
const char *Pa_GetVersionText(void);

typedef int PaDeviceIndex;

typedef enum PaHostApiTypeId {
    paInDevelopment=0,
    paDirectSound=1,
    paMME=2,
    paASIO=3,
    paSoundManager=4,
    paCoreAudio=5,
    paOSS=7,
    paALSA=8,
    paAL=9,
    paBeOS=10,
    paWDMKS=11,
    paJACK=12,
    paWASAPI=13,
    paAudioScienceHPI=14
} PaHostApiTypeId;

typedef struct PaHostApiInfo {
    int structVersion;
    PaHostApiTypeId type;
    const char *name;
    int deviceCount;
    PaDeviceIndex defaultInputDevice;
    PaDeviceIndex defaultOutputDevice;
} PaHostApiInfo;

typedef int PaHostApiIndex;

PaHostApiIndex Pa_GetHostApiCount();
const PaHostApiInfo *Pa_GetHostApiInfo(PaHostApiIndex);

typedef double PaTime;

typedef struct PaDeviceInfo {
    int structVersion;
    const char *name;
    PaHostApiIndex hostApi;
    int maxInputChannels;
    int maxOutputChannels;
    PaTime defaultLowInputLatency;
    PaTime defaultLowOutputLatency;
    PaTime defaultHighInputLatency;
    PaTime defaultHighOutputLatency;
    double defaultSampleRate;
} PaDeviceInfo;

PaDeviceIndex Pa_GetDeviceCount(void);
const PaDeviceInfo *Pa_GetDeviceInfo(PaDeviceIndex);

PaHostApiIndex Pa_GetDefaultHostApi(void);
PaDeviceIndex Pa_GetDefaultInputDevice(void);
PaDeviceIndex Pa_GetDefaultOutputDevice(void);

const char *Pa_GetErrorText(PaError);

typedef void PaStream;
typedef unsigned long PaSampleFormat;

typedef struct PaStreamParameters {
    PaDeviceIndex device;
    int channelCount;
    PaSampleFormat sampleFormat;
    PaTime suggestedLatency;
    void *hostApiSpecificStreamInfo;
} PaStreamParameters;

typedef unsigned long PaStreamFlags;

typedef struct PaStreamCallbackTimeInfo{
    PaTime inputBufferAdcTime;
    PaTime currentTime;
    PaTime outputBufferDacTime;
} PaStreamCallbackTimeInfo;

typedef unsigned long PaStreamCallbackFlags;

typedef int PaStreamCallback(const void*, void*, unsigned long,
                             const PaStreamCallbackTimeInfo*,
                             PaStreamCallbackFlags, void*);
typedef void PaStreamFinishedCallback(void*);

typedef struct PaStreamInfo {
    int structVersion;
    PaTime inputLatency;
    PaTime outputLatency;
    double sampleRate;
} PaStreamInfo;

PaError Pa_OpenStream(PaStream**, const PaStreamParameters*,
		      const PaStreamParameters*, double,
                      unsigned long, PaStreamFlags,
		      PaStreamCallback*, void*);
PaError Pa_CloseStream (PaStream*);
PaError Pa_SetStreamFinishedCallback(PaStream*, PaStreamFinishedCallback*);
PaError Pa_StartStream (PaStream*);
PaError Pa_StopStream (PaStream*);
PaError Pa_AbortStream (PaStream*);
PaError Pa_IsStreamStopped (PaStream*);
PaError Pa_IsStreamActive (PaStream*);
const PaStreamInfo *Pa_GetStreamInfo (PaStream*);
PaTime Pa_GetStreamTime (PaStream*);
double Pa_GetStreamCpuLoad (PaStream*);
PaError Pa_ReadStream (PaStream*, void*, unsigned long);
PaError Pa_WriteStream (PaStream*, const void*, unsigned long);
signed long Pa_GetStreamReadAvailable (PaStream*);
signed long Pa_GetStreamWriteAvailable (PaStream*);
PaError Pa_GetSampleSize (PaSampleFormat);
void Pa_Sleep (long);
""")

float32 = 0x00000001
int32   = 0x00000002
# int24   = 0x00000004 Not supported by numpy!
int16   = 0x00000008
int8    = 0x00000010
uint8   = 0x00000020

_pa2np = {
    float32: np.float32,
    int32: np.int32,
    # int24: None, Not supported by numpy!
    int16: np.int16,
    int8: np.int8,
    uint8: np.uint8
}

_npsizeof = {
    np.float32: 4,
    np.int32: 4,
    # np.int24: 3, Not supported by numpy!
    np.int16: 2,
    np.int8: 1,
    np.uint8: 1
}

_pa = ffi.dlopen('portaudio')
_pa.Pa_Initialize()

@atexit.register
def _terminate():
    _pa.Pa_Terminate()

def _api2dict(api, index):
    return { 'struct_version': api.structVersion,
             'type': api.type,
             'name': ffi.string(api.name),
             'api_idx': index,
             'device_count': api.deviceCount,
             'default_input_device_index': api.defaultInputDevice,
             'default_output_device_index': api.defaultOutputDevice }

def apis():
    for idx in range(_pa.Pa_GetHostApiCount()):
        yield _api2dict(_pa.Pa_GetHostApiInfo(idx), idx)


def _dev2dict(dev, index):
    return { 'struct_version': dev.structVersion,
             'name': ffi.string(dev.name),
             'device_index': index,
             'host_api_index': dev.hostApi,
             'input_channels': dev.maxInputChannels,
             'output_channels': dev.maxOutputChannels,
             'default_low_input_latency': dev.defaultLowInputLatency,
             'default_low_output_latency': dev.defaultLowOutputLatency,
             'default_high_input_latency': dev.defaultHighInputLatency,
             'default_high_output_latency': dev.defaultHighOutputLatency,
             'default_sample_rate': dev.defaultSampleRate,
             'input_latency': dev.defaultLowInputLatency,
             'output_latency': dev.defaultLowOutputLatency,
             'sample_format': float32,
             'interleaved_data': True }

def devices():
    for idx in range(_pa.Pa_GetDeviceCount()):
        yield _dev2dict(_pa.Pa_GetDeviceInfo(idx), idx)

def default_api():
    idx = _pa.Pa_GetDefaultHostApi()
    return _api2dict(_pa.Pa_GetHostApiInfo(idx), idx)

def default_input_device():
    idx = _pa.Pa_GetDefaultInputDevice()
    return _dev2dict(_pa.Pa_GetDeviceInfo(idx), idx)

def default_output_device():
    idx = _pa.Pa_GetDefaultOutputDevice()
    return _dev2dict(_pa.Pa_GetDeviceInfo(idx), idx)

def pa_version():
    return (_pa.Pa_GetVersion(), ffi.string(_pa.Pa_GetVersionText()))

class Stream(object):
    def __init__(self, sample_rate=44100, block_length=1024,
                 input_device=None, output_device=None,
                 callback=None, finished_callback=None,
                 **flags):
        if input_device is None:
            input_device = default_input_device()
        if output_device is None:
            output_device = default_output_device()
        if callback is not None or finished_callback is not None:
            raise NotImplementedError("stream callbacks are not implemented yet")

        input_stream_parameters = ffi.new("PaStreamParameters*",
                                          { 'device': input_device['device_index'],
                                            'channelCount': input_device['input_channels'],
                                            'sampleFormat': input_device['sample_format'],
                                            'suggestedLatency': input_device['input_latency'],
                                            'hostApiSpecificStreamInfo': ffi.NULL })
        self._input_format = _pa2np[input_stream_parameters.sampleFormat]
        self._input_channels = input_stream_parameters.channelCount
        if not input_device['interleaved_data']:
            input_stream_parameters.sampleFormat |= 0x80000000

        output_stream_parameters = ffi.new("PaStreamParameters*",
                                           { 'device': output_device['device_index'],
                                            'channelCount': output_device['output_channels'],
                                            'sampleFormat': output_device['sample_format'],
                                            'suggestedLatency': output_device['output_latency'],
                                             'hostApiSpecificStreamInfo': ffi.NULL })
        self._output_format = _pa2np[output_stream_parameters.sampleFormat]
        self._output_channels = output_stream_parameters.channelCount
        if not output_device['interleaved_data']:
            output_stream_parameters.sampleFormat |= 0x80000000

        stream_flags = 0
        if 'no_clipping' in flags:
            stream_flags |= 0x00000001
        if 'no_dithering' in flags:
            stream_flags |= 0x00000002
        if 'never_drop_input' in flags and flags['never_drop_input']:
            stream_flags |= 0x00000004
        if 'prime_output_buffers_using_callback' in flags:
            stream_flags |= 0x00000008

        self._stream = ffi.new("PaStream**")
        err = _pa.Pa_OpenStream(self._stream,
                                input_stream_parameters, output_stream_parameters,
                                sample_rate, block_length, stream_flags, ffi.NULL, ffi.NULL)
        self._handle_error(err)

    def _handle_error(self, err):
        if err >= 0: return err
        print("Warning (%.4f): %s" % (self.time(), ffi.string(_pa.Pa_GetErrorText(err))))

    def __del__(self):
        # At program shutdown, _pa is sometimes deleted before this
        # function is called. However, in that case, Pa_Terminate
        # already took care of closing all dangling streams.
        if _pa:
            self._handle_error(_pa.Pa_CloseStream(self._stream[0]))

    def start(self):
        self._handle_error(_pa.Pa_StartStream(self._stream[0]))

    def stop(self):
        self._handle_error(_pa.Pa_StopStream(self._stream[0]))

    def abort(self):
        self._handle_error(_pa.Pa_AbortStream(self._stream[0]))

    def is_active(self):
        return self._handle_error(_pa.Pa_IsStreamActive(self._stream[0])) == 1

    def is_stopped(self):
        return self._handle_error(_pa.Pa_IsStreamStopped(self._stream[0])) == 1

    def read_length(self):
        return _pa.Pa_GetStreamReadAvailable(self._stream[0])

    def write_length(self):
        return _pa.Pa_GetStreamWriteAvailable(self._stream[0])

    def info(self):
        info = _pa.Pa_GetStreamInfo(self._stream[0])
        return { 'struct_version': info.structVersion,
                 'input_latency': info.inputLatency,
                 'output_latency': info.outputLatency,
                 'sample_rate': info.sampleRate }

    def time(self):
        return _pa.Pa_GetStreamTime(self._stream[0])

    def cpu_load(self):
        return _pa.Pa_GetStreamCpuLoad(self._stream[0])

    def read(self, num_frames=1024, raw=False):
        num_bytes = self._input_channels*_npsizeof[self._input_format]*num_frames
        data = ffi.new("char[]", num_bytes)
        self._handle_error(_pa.Pa_ReadStream(self._stream[0], data, num_frames))
        if raw:
            return data
        else:
            data = np.fromstring(ffi.buffer(data), dtype=self._input_format,
                                 count=num_frames*self._input_channels)
            return np.reshape(data, (num_frames, self._input_channels))

    def write(self, data, num_frames=None):
        num_frames = num_frames or len(data)
        if isinstance(data, np.ndarray):
            if data.dtype != self._output_format:
                data = np.array(data, dtype=self._output_format)
            data = data.flatten().tostring()
        elif isinstance(data, list):
            data = np.array(data, dtype=self._output_format).flatten().tostring()
        self._handle_error(_pa.Pa_WriteStream(self._stream[0], data, num_frames))

if __name__ == '__main__':
    from scipy.io.wavfile import read as wavread
    fs, wave = wavread('thistle.wav')
    wave = np.array(wave, dtype=np.float32)
    wave /= 2**15
    block_length = 1
    s = Stream(sample_rate=fs, block_length=block_length)
    s.start()
    for n in range(int(fs*5/block_length)):
        s.write(s.read(block_length))
    for idx in range(0, wave.size, block_length):
        s.write(wave[idx:idx+block_length])
    s.stop()
