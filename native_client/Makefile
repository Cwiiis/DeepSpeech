
TFDIR ?= ../../tensorflow

default: deepspeech

clean:
	rm -f kiss_fft.o kiss_fftr.o c_speech_features.o client.o deepspeech

kiss_fft.o: kiss_fft130/kiss_fft.c kiss_fft130/kiss_fft.h kiss_fft130/_kiss_fft_guts.h
	cc -c kiss_fft130/kiss_fft.c -lm

kiss_fftr.o: kiss_fft130/kiss_fft.c kiss_fft130/kiss_fft.h kiss_fft130/_kiss_fft_guts.h kiss_fft130/tools/kiss_fftr.c kiss_fft130/tools/kiss_fftr.h
	cc -c kiss_fft130/tools/kiss_fftr.c -lm -Ikiss_fft130

c_speech_features.o: c_speech_features/c_speech_features.h c_speech_features/c_speech_features.c
	cc -c c_speech_features/c_speech_features.c -lm -Ikiss_fft130 -Ic_speech_features

client.o: client.cc
	c++ -c client.cc `pkg-config --cflags sox` -Ikiss_fft130 -Ic_speech_features

deepspeech: kiss_fft.o kiss_fftr.o c_speech_features.o client.o
	c++ -o deepspeech client.o kiss_fft.o kiss_fftr.o c_speech_features.o `pkg-config --libs sox` -L${TFDIR}/bazel-bin/tensorflow -L${TFDIR}/bazel-bin/native_client -ldeepspeech -ltensorflow

run: deepspeech
	LD_LIBRARY_PATH=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${LD_LIBRARY_PATH} ./deepspeech ${ARGS}
