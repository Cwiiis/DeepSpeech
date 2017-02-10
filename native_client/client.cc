#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sox.h>
#include "deepspeech.h"
#include "c_speech_features.h"

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

#define INPUT_SIZE (N_CEP + (2 * N_CEP * N_CONTEXT))
#define CONTEXT_SIZE (N_CEP * N_CONTEXT)

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

  // Compute MFCC features
  float** mfcc;
  int n_frames = csf_mfcc((const short*)buffer, buffer_size / 2, SAMPLE_RATE,
                          WIN_LEN, WIN_STEP, N_CEP, N_FILTERS, N_FFT,
                          LOWFREQ, HIGHFREQ, 0.97f, 22, 1, &mfcc);

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

  // Free mfcc array
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
