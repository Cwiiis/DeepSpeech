from datetime import datetime
from deepspeech import audioToInputVector
from python_speech_features import mfcc as psf_mfcc
from c_speech_features import mfcc as csf_mfcc
from util.audio import audiofile_to_input_vector
import scipy.io.wavfile as wav

speed_test_iterations = 500

def ds_audiofile_to_input_vector(audio_filename, ncep, ncontext):
  fs, audio = wav.read(audio_filename)
  return audioToInputVector(audio, fs, ncep, ncontext)

def time(callback):
    before = datetime.utcnow()
    for i in range(speed_test_iterations):
        callback()
    return (datetime.utcnow() - before).total_seconds()

time1 = time(lambda: audiofile_to_input_vector('data/ldc93s1/LDC93S1.wav', 26, 9))
time2 = time(lambda: ds_audiofile_to_input_vector('data/ldc93s1/LDC93S1.wav', 26, 9))
print 'audiofile_to_input_vector: %.3fx' % (time1 / time2)




from deepspeech import mfcc as ds_mfcc

fs, audio = wav.read('data/ldc93s1/LDC93S1.wav')
time1 = time(lambda: ds_mfcc(audio, samplerate=fs, numcep=26))
time2 = time(lambda: csf_mfcc(audio, samplerate=fs, numcep=26))
time3 = time(lambda: psf_mfcc(audio, samplerate=fs, numcep=26))
print 'DS: %.2fs, CSF: %.2f, PSF: %.2f' % (time1, time2, time3)
print 'DS: %.2fx, CSF: %.2fx' % (time3/time1, time3/time2)
