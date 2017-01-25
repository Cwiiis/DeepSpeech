
TFDIR ?= ../../tensorflow

libmfcc.o: libmfcc/libmfcc.c libmfcc/libmfcc.h
	c++ -c libmfcc/libmfcc.c

client.o: client.cc
	c++ -c client.cc `pkg-config --cflags sox`

deepspeech: libmfcc.o client.o
	g++ -o deepspeech client.o libmfcc.o `pkg-config --libs sox` -L${TFDIR}/bazel-bin/tensorflow -L${TFDIR}/bazel-bin/native_client -ldeepspeech -ltensorflow

run: deepspeech
	LD_LIBRARY_PATH=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${LD_LIBRARY_PATH} ./deepspeech ${ARGS}
