#include <assert.h>
#include <float.h>
#include <math.h>
#include <sox.h>
#include "deepspeech.h"
#include "tools/kiss_fftr.h"

#define SAMPLE_RATE 16000
#define COEFF 0.97f
#define WIN_LEN 0.025f
#define WIN_STEP 0.01f
#define N_FFT 512
#define N_FILTERS 26
#define LOWFREQ 0
#define HIGHFREQ (SAMPLE_RATE/2)
#define N_CEP 26
#define CEP_LIFTER 22
#define N_CONTEXT 9

#define FFT_OUT (N_FFT / 2 + 1)
#define INPUT_SIZE (N_CEP + (2 * N_CEP * N_CONTEXT))
#define CONTEXT_SIZE (N_CEP * N_CONTEXT)

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define HZ2MEL(x) (2595 * log10f(1+x/700.0f))
#define MEL2HZ(x) (700 * (powf(10.0f, x/2595.0f) - 1))

int main(int argc, char **argv)
{
  if (argc < 3) {
    printf("Usage: deepspeech [model path] [audio path]\n");
    return 1;
  }

  // Initialise DeepSpeech
  DeepSpeechContext* ctx = DsInit(argv[1]);
  assert(ctx);

  // Initialise SOX
  assert(sox_init() == SOX_SUCCESS);

  sox_format_t* input = sox_open_read(argv[2], NULL, NULL, NULL);
  assert(input);

  // Resample/reformat the audio so we can pass it through the MFCC functions
  sox_signalinfo_t target_signal = {
      SAMPLE_RATE, // Rate
      1, // Channels
      16, // Precision
      SOX_UNSPEC, // Length
      NULL // Effects headroom multiplier
  };

  sox_encodinginfo_t target_encoding = {
    SOX_ENCODING_SIGN2, // Sample format
    16, // Bits per sample
    0.0, // Compression factor
    sox_option_default, // Should bytes be reversed
    sox_option_default, // Should nibbles be reversed
    sox_option_default, // Should bits be reversed (?!)
    sox_false // Reverse endianness
  };

  char *buffer;
  size_t buffer_size;
  sox_format_t* output = sox_open_memstream_write(&buffer, &buffer_size,
                                                  &target_signal,
                                                  &target_encoding,
                                                  "raw", NULL);
  assert(output);

  // Setup the effects chain to decode/resample
  char* sox_args[10];
  sox_effects_chain_t* chain =
    sox_create_effects_chain(&input->encoding, &output->encoding);

  sox_effect_t* e = sox_create_effect(sox_find_effect("input"));
  sox_args[0] = (char*)input;
  assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &input->signal, &input->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("channels"));
  assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &input->signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("rate"));
  assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &input->signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("output"));
  sox_args[0] = (char*)output;
  assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &input->signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  // Finally run the effects chain
  sox_flow_effects(chain, NULL, NULL);
  sox_delete_effects_chain(chain);
  sox_close(output);

  // Calculate frame variables and padding
  int slen = buffer_size / 2;
  int frame_len = (int)roundf(WIN_LEN * SAMPLE_RATE);
  int frame_step = (int)roundf(WIN_STEP * SAMPLE_RATE);
  int n_frames = 1;
  if (slen > frame_len) {
    n_frames = 1 + (int)ceilf((slen - frame_len) / (float)frame_step);
  }
  int padlen = (n_frames - 1) * frame_step + frame_len;

  // Preemphasis
  float* buffer_preemph = (float*)calloc(sizeof(float), slen + padlen);
  short* sbuffer = (short*)buffer;
  for (int i = slen - 1; i >= 1; i--) {
    buffer_preemph[i] = sbuffer[i] - sbuffer[i-1] * COEFF;
  }
  buffer_preemph[0] = (float)sbuffer[0];
  free(buffer);

  // Frame into overlapping frames
  int** indices = (int**)malloc(sizeof(int*) * n_frames);
  for (int i = 0; i < n_frames; i++) {
    indices[i] = (int*)malloc(sizeof(int) * frame_len);
    int base = i * frame_step;
    for (int j = 0; j < frame_len; j++) {
      indices[i][j] = base + j;
    }
  }

  float** frames = (float**)malloc(sizeof(float*) * n_frames);
  for (int i = 0; i < n_frames; i++) {
    frames[i] = (float*)calloc(sizeof(float), MAX(N_FFT, frame_len));
    for (int j = 0; j < frame_len; j++) {
      frames[i][j] = buffer_preemph[indices[i][j]];
    }
    free (indices[i]);
  }
  free(indices);
  free(buffer_preemph);

  // Compute the power spectrum of each frame
  kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, NULL, NULL);
  kiss_fft_cpx out[FFT_OUT];
  float** pspec = (float**)malloc(sizeof(float*) * n_frames);
  for (int i = 0; i < n_frames; i++) {
    pspec[i] = (float*)malloc(sizeof(float) * FFT_OUT);
    // Compute the magnitude spectrum
    kiss_fftr(cfg, frames[i], out);
    free(frames[i]);
    for (int j = 0; j < FFT_OUT; j++) {
      // Compute the power spectrum
      float abs = sqrtf(pow(out[j].r, 2.0f) + pow(out[j].i, 2.0f));
      pspec[i][j] = (1.0/N_FFT) * powf(abs, 2.0f);
    }
  }
  free(frames);

  // Compute the energy
  float* energy = (float*)calloc(sizeof(float), n_frames);
  for (int i = 0; i < n_frames; i++) {
    for (int j = 0; j < FFT_OUT; j++) {
      energy[i] += pspec[i][j];
    }
    if (energy[i] == 0.0f) {
      energy[i] = FLT_MIN;
    }
  }

  // Compute a Mel-filterbank
  float lowmel = HZ2MEL(LOWFREQ);
  float highmel = HZ2MEL(HIGHFREQ);
  int bin[N_FILTERS + 2];

  for (int i = 0; i < N_FILTERS + 2; i++) {
    float melpoint = ((highmel - lowmel) / (float)(N_FILTERS + 1) * i) + lowmel;
    bin[i] = (int)floorf((N_FFT + 1) * MEL2HZ(melpoint) / (float)SAMPLE_RATE);
  }

  float fbank[N_FILTERS][FFT_OUT] = { 0, };
  for (int i = 0; i < N_FILTERS; i++) {
    int start = MIN(bin[i], bin[i+1]);
    int end = MAX(bin[i], bin[i+1]);
    for (int j = start; j < end; j++) {
      fbank[i][j] = (j - bin[i]) / (float)(bin[i+1]-bin[i]);
    }
    start = MIN(bin[i+1], bin[i+2]);
    end = MAX(bin[i+1], bin[i+2]);
    for (int j = start; j < end; j++) {
      fbank[i][j] = (bin[i+2]-j) / (float)(bin[i+2]-bin[i+1]);
    }
  }

  // Compute the filter-bank energies
  float** feat = (float**)malloc(sizeof(float*) * n_frames);
  for (int i = 0; i < n_frames; i++) {
    feat[i] = (float*)calloc(sizeof(float), N_FILTERS);
    for (int j = 0; j < N_FILTERS; j++) {
      for (int k = 0; k < FFT_OUT; k++) {
        feat[i][j] += pspec[i][k] * fbank[j][k];
      }
      if (feat[i][j] == 0.0f) {
        feat[i][j] = FLT_MIN;
      }
      feat[i][j] = logf(feat[i][j]);
    }
  }

  // Perform DCT-II
  float sf1 = sqrtf(1 / (4 * (float)N_FILTERS));
  float sf2 = sqrtf(1 / (2 * (float)N_FILTERS));
  float** mfcc = (float**)malloc(sizeof(float*) * n_frames);
  for (int i = 0; i < n_frames; i++) {
    mfcc[i] = (float*)calloc(sizeof(float), N_CEP);
    for (int j = 0; j < N_CEP; j++) {
      for (int k = 0; k < N_CEP; k++) {
        mfcc[i][j] += feat[i][k] *
          cosf(M_PI * j * (2 * k + 1) / (2 * N_FILTERS));
      }
      mfcc[i][j] *= 2 * ((i == 0 && j == 0) ? sf1 : sf2);
    }
  }

  for (int i = 0; i < n_frames; i++) {
    free(feat[i]);
  }
  free(feat);

  // Apply a cepstral lifter
  for (int i = 0; i < n_frames; i++) {
    for (int j = 0; j < N_CEP; j++) {
      mfcc[i][j] *= 1 + (CEP_LIFTER / 2.0f) * sinf(M_PI * j / CEP_LIFTER);
    }
  }

  // Append energies
  for (int i = 0; i < n_frames; i++) {
    mfcc[i][0] = logf(energy[i]);
  }
  free(energy);

  // mfcc now contains the equivalent of python_speech_features.mfcc

  // Take every other frame (BiRNN stride of 2) and add past/future context
  int ds_input_length = (n_frames + 1) / 2;
  float** ds_input = (float**)malloc(sizeof(float*) * ds_input_length);
  for (int i = 0; i < ds_input_length; i++) {
    // TODO: Use MFCC of silence instead of zero
    ds_input[i] = (float*)calloc(sizeof(float), INPUT_SIZE);

    // Past context
    for (int j = N_CONTEXT; j > 0; j--) {
      int frame_index = (i * 2) - (j * 2);
      if (frame_index < 0) { continue; }
      int base = (N_CONTEXT - j) * N_CEP;
      for (int k = 0; k < N_CEP; k++) {
        ds_input[i][base + k] = mfcc[frame_index][k];
      }
    }

    // Present context
    for (int j = 0; j < N_CEP; j++) {
      ds_input[i][j + CONTEXT_SIZE] = mfcc[i * 2][j];
    }

    // Future context
    for (int j = 1; j <= N_CONTEXT; j++) {
      int frame_index = (i * 2) + (j * 2);
      if (frame_index >= n_frames) { continue; }
      int base = CONTEXT_SIZE + N_CEP + ((j - 1) * N_CEP);
      for (int k = 0; k < N_CEP; k++) {
        ds_input[i][base + k] = mfcc[frame_index][k];
      }
    }
  }

  // Free mfcc process
  for (int i = 0; i < n_frames; i++) {
    free(mfcc[i]);
  }
  free(mfcc);

  // Whiten inputs (TODO: Should we whiten)
  double n_inputs = (double)(ds_input_length * INPUT_SIZE);
  double mean = 0.0;
  for (int i = 0; i < ds_input_length; i++) {
    for (int j = 0; j < INPUT_SIZE; j++) {
      mean += ds_input[i][j] / n_inputs;
    }
  }

  double stddev = 0.0;
  for (int i = 0; i < ds_input_length; i++) {
    for (int j = 0; j < INPUT_SIZE; j++) {
      stddev += pow(fabs(ds_input[i][j] - mean), 2.0) / n_inputs;
    }
  }
  stddev = sqrt(stddev);

  for (int i = 0; i < ds_input_length; i++) {
    for (int j = 0; j < INPUT_SIZE; j++) {
      ds_input[i][j] = (float)((ds_input[i][j] - mean) / stddev);
    }
  }

  // Pass prepared audio to DeepSpeech
  // ...

  // Free memory
  for (int i = 0; i < ds_input_length; i++) {
    free(ds_input[i]);
  }
  free(ds_input);

  // Deinitialise and quit
  DsClose(ctx);
  sox_quit();

  return 0;
}
