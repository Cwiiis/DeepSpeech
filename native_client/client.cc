#include <assert.h>
#include <sox.h>
#include "deepspeech.h"
#include "kiss_fft.h"
#include "libmfcc.h"

#define WIN_SIZE 512

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
      16000, // Rate
      1, // Channels
      16, // Precision
      SOX_UNSPEC, // Length
      NULL // Effects headroom multiplier
  };

  sox_encodinginfo_t target_encoding = {
    SOX_ENCODING_UNSIGNED, // Sample format
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

  e = sox_create_effect(sox_find_effect("rate"));
  assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &input->signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("channels"));
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

  // Run buffer through FFT
  kiss_fft_cpx in[WIN_SIZE];
  size_t nfft = buffer_size / (WIN_SIZE * 2);
  kiss_fft_cpx* out =
    (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft * WIN_SIZE);
  kiss_fft_cfg cfg = kiss_fft_alloc(WIN_SIZE, 0, 0, 0);

  for (int i = 0; i < nfft; i++) {
    int buffer_index = i * WIN_SIZE;
    for (int j = buffer_index; j < buffer_index + WIN_SIZE; j++) {
      short value = ((short*)buffer)[j];
      in[j - buffer_index].r = (kiss_fft_scalar)((double)value / 32768.0);
      in[j - buffer_index].i = 0.0;
    }
    kiss_fft(cfg, in, &out[i * WIN_SIZE]);
  }
  free(buffer);

  // Convert complex values to amplitude
  // ...

  // Run FFT through MFCC
  // ...

  // Pass MFCC to DeepSpeech
  // ...

  // Deinitialise and quit
  DsClose(ctx);
  sox_quit();

  return 0;
}
