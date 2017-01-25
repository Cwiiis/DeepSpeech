#include <iostream>
#include <sox.h>
#include "deepspeech.h"

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::cout << "Usage: deepspeech [model path] [audio path]\n";
    return 1;
  }

  // Initialise DeepSpeech
  DeepSpeechContext* ctx = DsInit(argv[1]);
  if (!ctx) {
    std::cout << "Error initialising DeepSpeech.\n";
    return 1;
  }

  // Initialise SOX
  int sox_return_code = sox_format_init();
  if (sox_return_code != 0) {
    std::cout << "Error initialising SoX\n";
    return 1;
  }

  sox_format_t* audio_file = sox_open_read(argv[2], NULL, NULL, NULL);
  if (audio_file) {
    sox_sample_t *buffer =
      static_cast<sox_sample_t *>(malloc(sizeof(sox_sample_t) * audio_file->olength));
    size_t samples = sox_read(audio_file, buffer, audio_file->olength);

    if (samples > 0) {
      
    } else {
      std::cout << "Error reading audio file\n";
    }

    free(buffer);
  } else {
    std::cout << "Error opening audio file\n";
  }

  DsClose(ctx);
  sox_format_quit();

  return 0;
}
