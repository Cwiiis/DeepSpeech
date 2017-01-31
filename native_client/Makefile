
TFDIR ?= ../../tensorflow

clean:
	rm -f libmfcc.o kiss_fft.o kiss_fftr.o client.o deepspeech

libmfcc.o: libmfcc/libmfcc.c libmfcc/libmfcc.h
	c++ -c libmfcc/libmfcc.c

kiss_fft.o: kiss_fft130/kiss_fft.c kiss_fft130/kiss_fft.h kiss_fft130/_kiss_fft_guts.h
	cc -c kiss_fft130/kiss_fft.c -lm

kiss_fftr.o: kiss_fft130/kiss_fft.c kiss_fft130/kiss_fft.h kiss_fft130/_kiss_fft_guts.h kiss_fft130/tools/kiss_fftr.c kiss_fft130/tools/kiss_fftr.h
	cc -c kiss_fft130/tools/kiss_fftr.c -lm -Ikiss_fft130

client.o: client.cc
	c++ -c client.cc `pkg-config --cflags sox` -Ikiss_fft130 -Ilibmfcc

deepspeech: libmfcc.o kiss_fft.o kiss_fftr.o client.o
	c++ -o deepspeech client.o libmfcc.o kiss_fft.o kiss_fftr.o `pkg-config --libs sox` -L${TFDIR}/bazel-bin/tensorflow -L${TFDIR}/bazel-bin/native_client -ldeepspeech -ltensorflow

run: deepspeech
	LD_LIBRARY_PATH=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${LD_LIBRARY_PATH} ./deepspeech ${ARGS}
