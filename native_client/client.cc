#include <assert.h>
#include <float.h>
#include <math.h>
#include <sox.h>
#include "deepspeech.h"
#include "tools/kiss_fftr.h"
#include "libmfcc.h"

#define SAMPLE_RATE 16000
#define COEFF 0.97f
#define WIN_LEN 0.025f
#define WIN_STEP 0.01f
#define NFFT 512
#define FFT_OUT (NFFT / 2 + 1)
#define N_FILTERS 26
#define LOWFREQ 0
#define HIGHFREQ (SAMPLE_RATE/2)

#define N_CEPSTRUM 26

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

  printf("Buffer length: %d\n", buffer_size);

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

  printf("Audio[0..5]     = (%hhd, %hhd),\n"
         "                  (%hhd, %hhd),\n"
         "                  (%hhd, %hhd)\n",
         buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5]);
  printf("sAudio[0..2]    = %hd, %hd, %hd\n",
         sbuffer[0], sbuffer[1], sbuffer[2]);
  free(buffer);

  // Frame into overlapping frames
  /*
  indices =
    numpy.tile(numpy.arange(0,frame_len),
               (numframes,1)) +
    numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),
               (frame_len,1)).T
  */
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
    frames[i] = (float*)calloc(sizeof(float), MAX(NFFT, frame_len));
    for (int j = 0; j < frame_len; j++) {
      frames[i][j] = buffer_preemph[indices[i][j]];
    }
    free (indices[i]);
  }
  free(indices);

  printf("n_frames = %d, frame_len = %d\n", n_frames, frame_len);

  printf("PreAudio[0..2]  = %.2f, %.2f, %.2f\n"
         "Frames[0][0..2] = %.2f, %.2f, %.2f\n",
         buffer_preemph[0], buffer_preemph[1], buffer_preemph[2],
         frames[0][0], frames[0][1], frames[0][2]);
  free(buffer_preemph);

  // Compute the power spectrum of each frame
  kiss_fftr_cfg cfg = kiss_fftr_alloc(NFFT, 0, NULL, NULL);
  kiss_fft_cpx out[FFT_OUT];
  float** pspec = (float**)malloc(sizeof(float*) * n_frames);
  for (int i = 0; i < n_frames; i++) {
    pspec[i] = (float*)malloc(sizeof(float) * FFT_OUT);
    // Compute the magnitude spectrum
    kiss_fftr(cfg, frames[i], out);
    for (int j = 0; j < FFT_OUT; j++) {
      // Compute the power spectrum
      float abs = sqrtf(pow(out[j].r, 2.0f) + pow(out[j].i, 2.0f));
      pspec[i][j] = (1.0/NFFT) * powf(abs, 2.0f);
    }
  }

  printf("pspec[0][0..2] = %f, %f, %f\n",
         pspec[0][0], pspec[0][1], pspec[0][2]);

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

  printf("energy[0..2] = %f, %f, %f\n", energy[0], energy[1], energy[2]);

  // Compute a Mel-filterbank
  float lowmel = HZ2MEL(LOWFREQ);
  float highmel = HZ2MEL(HIGHFREQ);
  int bin[N_FILTERS + 2];

  for (int i = 0; i < N_FILTERS + 2; i++) {
    float melpoint = ((highmel - lowmel) / (float)(N_FILTERS + 1) * i) + lowmel;
    bin[i] = (int)floorf((NFFT + 1) * MEL2HZ(melpoint) / (float)SAMPLE_RATE);
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

  printf("fbank[0][0..2] = %.2f, %.2f, %.2f\n"
         "fbank[n][n-2..n] = %.2f, %.2f, %.2f\n",
         fbank[0][0], fbank[0][1], fbank[0][2],
         fbank[N_FILTERS-1][FFT_OUT-3],
         fbank[N_FILTERS-1][FFT_OUT-2],
         fbank[N_FILTERS-1][FFT_OUT-1]);

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

  printf("feat[0][0..2] = %.4f, %.4f, %.4f\n"
         "feat[n][n-2..n] = %.4f, %.4f, %.4f\n",
         feat[0][0], feat[0][1], feat[0][2],
         feat[n_frames-1][N_FILTERS-3],
         feat[n_frames-1][N_FILTERS-2],
         feat[n_frames-1][N_FILTERS-1]);

  // Perform DCT-II

  // Apply a cepstral lifter

  // Append energies

  /*
  // Run buffer through FFT
  size_t features = buffer_size / (WIN_SIZE * 2);
  kiss_fft_scalar in[WIN_SIZE];
  kiss_fft_cpx out[WIN_SIZE / 2 + 1];
  kiss_fftr_cfg cfg = kiss_fftr_alloc(WIN_SIZE, 0, NULL, NULL);
  size_t out_real_size = features * (WIN_SIZE / 2 + 1);
  double *out_real = (double*)malloc(sizeof(double) * out_real_size);

  for (int i = 0; i < features; i++) {
    int buffer_index = i * WIN_SIZE;
    for (int j = buffer_index; j < buffer_index + WIN_SIZE; j++) {
      short value = ((short*)buffer)[j];
      in[j - buffer_index] = (kiss_fft_scalar)((double)value / 32768.0);
    }
    kiss_fftr(cfg, in, out);

    buffer_index = i * (WIN_SIZE / 2 + 1);
    for (int j = 0; j < WIN_SIZE / 2 + 1; j++) {
      out_real[j + buffer_index] = out[j].r;
    }
  }
  free(buffer);

  printf("Real features: %d\n", out_real_size);

  // Run FFT through MFCC, with a 0.01ms step and a 0.025ms window
  size_t out_mfcc_size =
    floor(((buffer_size / 2.0) / (SAMPLE_RATE / 1000.0)) / WIN_STEP);
  double** out_mfcc = (double**)malloc(sizeof(double*) * out_mfcc_size);
  for (int i = 0; i < out_mfcc_size; i++) {
    out_mfcc[i] = (double*)malloc(sizeof(double) * N_CEPSTRUM);

    for (int j = 0; j < N_CEPSTRUM; j++) {
      out_mfcc[i][j] =
        GetCoefficient(&out_real[(out_real_size - BIN_SIZE)/out_mfcc_size * i],
                       SAMPLE_RATE, N_FILTERS, BIN_SIZE, j);
    }
  }
  free(out_real);

  printf("Some coefficients:\n"
         "%lf, %lf, %lf, %lf, %lf\n"
         "%lf, %lf, %lf, %lf, %lf\n"
         "%lf, %lf, %lf, %lf, %lf\n"
         "%lf, %lf, %lf, %lf, %lf\n"
         "%lf\n",
         out_mfcc[1][0],
         out_mfcc[1][1],
         out_mfcc[1][2],
         out_mfcc[1][3],
         out_mfcc[1][4],
         out_mfcc[1][5],
         out_mfcc[1][6],
         out_mfcc[1][7],
         out_mfcc[1][8],
         out_mfcc[1][9],
         out_mfcc[1][10],
         out_mfcc[1][11],
         out_mfcc[1][12],
         out_mfcc[1][13],
         out_mfcc[1][14],
         out_mfcc[1][15],
         out_mfcc[1][16],
         out_mfcc[1][17],
         out_mfcc[1][18],
         out_mfcc[1][19],
         out_mfcc[1][20],
         out_mfcc[1][21],
         out_mfcc[1][22],
         out_mfcc[1][23],
         out_mfcc[1][24],
         out_mfcc[1][25]);

  // Pass MFCC to DeepSpeech
  // ...
  for (int i = 0; i < out_mfcc_size; i++) {
    free(out_mfcc[i]);
  }
  free(out_mfcc);*/

  // Deinitialise and quit
  DsClose(ctx);
  sox_quit();

  return 0;
}
