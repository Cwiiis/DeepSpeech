%module deepspeech

%{
#define SWIG_FILE_WITH_INIT
#include "deepspeech.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (short* IN_ARRAY1, int DIM1) {(const short* aBuffer, unsigned int aBufferSize)};
%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(float** aMfcc, int* aNFrames, int* aFrameLen)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* aMfcc, int aNFrames, int aFrameLen)};


%apply (short* IN_ARRAY1, int DIM1) {(const short* signal, unsigned int signal_len)};
%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(float** mfcc, int* mfcc_dim1, int* mfcc_dim2)};
%typemap(default) int samplerate { $1 = 16000; }
%typemap(default) float winlen { $1 = 0.025; }
%typemap(default) float winstep { $1 = 0.01; }
%typemap(default) int numcep { $1 = 13; }
%typemap(default) int nfilt { $1 = 26; }
%typemap(default) int nfft { $1 = 512; }
%typemap(default) int lowfreq { $1 = 0; }
%typemap(default) int highfreq { $1 = 0; }
%typemap(default) float preemph { $1 = 0.97; }
%typemap(default) int ceplifter { $1 = 22; }
%typemap(default) int appendEnergy { $1 = 1; }
%typemap(default) float* winfunc { $1 = NULL; }


%include "../deepspeech.h"
