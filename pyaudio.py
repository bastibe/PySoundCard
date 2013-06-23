import os
os.environ['DYLD_LIBRARY_PATH'] = '/opt/local/lib'
from cffi import FFI

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
""")
C = ffi.dlopen('portaudio')

class pyaudio:
    def __init__(self):
        self._pa = ffi.dlopen('portaudio')
        self._pa.Pa_Initialize()

        def api2dict(api, index):
            return { 'structVersion': api.structVersion,
                     'type': api.type,
                     'name': ffi.string(api.name),
                     'apiIdx': index,
                     'deviceCount': api.deviceCount,
                     'defaultInputDevice': api.defaultInputDevice,
                     'defaultOutputDevice': api.defaultOutputDevice }

        self.apis = [api2dict(self._pa.Pa_GetHostApiInfo(idx), idx)
                     for idx in range(self._pa.Pa_GetHostApiCount())]

        def dev2dict(dev, index):
            return { 'structVersion': dev.structVersion,
                     'name': ffi.string(dev.name),
                     'deviceIndex': index,
                     'hostApi': dev.hostApi,
                     'maxInputChannels': dev.maxInputChannels,
                     'maxOutputChannels': dev.maxOutputChannels,
                     'defaultLowInputLatency': dev.defaultLowInputLatency,
                     'defaultLowOutputLatency': dev.defaultLowOutputLatency,
                     'defaultHighInputLatency': dev.defaultHighInputLatency,
                     'defaultHighOutputLatency': dev.defaultHighOutputLatency,
                     'defaultSampleRate': dev.defaultSampleRate }

        self.devices = [dev2dict(self._pa.Pa_GetDeviceInfo(idx), idx)
                        for idx in range(self._pa.Pa_GetDeviceCount())]

        for dev in self.devices:
            dev['hostApi'] = self.apis[dev['hostApi']]
        for api in self.apis:
            api['defaultInputDevice'] = self.devices[api['defaultInputDevice']]
            api['defaultOutputDevice'] = self.devices[api['defaultOutputDevice']]

        self.default_api = self.apis[self._pa.Pa_GetDefaultHostApi()]
        self.default_input_device = self.devices[self._pa.Pa_GetDefaultInputDevice()]
        self.default_output_device = self.devices[self._pa.Pa_GetDefaultOutputDevice()]
        self.pa_version = (self._pa.Pa_GetVersion(),
                           ffi.string(self._pa.Pa_GetVersionText()))

    def __del__(self):
        self._pa.Pa_Terminate()

if __name__ == '__main__':
    p = pyaudio()
    print(p.pa_version)
    for api in p.apis:
        print(api)
    for dev in p.devices:
        print(dev)
