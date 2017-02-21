
TFDIR ?= ../../tensorflow
TFMAKEFILEDIR ?= ${TFDIR}/tensorflow/contrib/makefile

default: client

clean:
	rm -f deepspeech deepspeech.o c_speech_features.o kiss_fft.o kiss_fftr.o libdeepspeech.so

client: client.cc
	c++ -o deepspeech client.cc `pkg-config --cflags --libs sox` -L${TFDIR}/bazel-bin/tensorflow -L${TFDIR}/bazel-bin/native_client -ldeepspeech -ltensorflow

client-libdeepspeech: libdeepspeech client.cc
	c++ -o deepspeech client.cc -L. -ldeepspeech `pkg-config --cflags --libs sox` ${TFMAKEFILEDIR}/gen/protobuf/lib/libprotobuf.a

libdeepspeech: ${TFDIR}/tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a kiss_fft130/kiss_fft.c kiss_fft130/kiss_fft.h kiss_fft130/_kiss_fft_guts.h kiss_fft130/tools/kiss_fftr.c kiss_fft130/tools/kiss_fftr.h c_speech_features/c_speech_features.c c_speech_features/c_speech_features.h deepspeech.cc deepspeech.h
	c++ -c -fPIC deepspeech.cc c_speech_features/c_speech_features.c kiss_fft130/kiss_fft.c kiss_fft130/tools/kiss_fftr.c -Ic_speech_features -Ikiss_fft130 -I${TFDIR} -I${TFMAKEFILEDIR}/gen/proto -I${TFMAKEFILEDIR}/downloads/protobuf/src -I${TFMAKEFILEDIR}/downloads/eigen
	c++ -shared -Wl,-soname,libdeepspeech.so -o libdeepspeech.so -Wl,--whole-archive ${TFMAKEFILEDIR}/gen/lib/libtensorflow-core.a -Wl,--no-whole-archive kiss_fft.o kiss_fftr.o c_speech_features.o deepspeech.o

run: client
	LD_LIBRARY_PATH=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${LD_LIBRARY_PATH} ./deepspeech ${ARGS}

debug: client
	LD_LIBRARY_PATH=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${LD_LIBRARY_PATH} gdb --args ./deepspeech ${ARGS}

run-libdeepspeech: client-libdeepspeech
	LD_LIBRARY_PATH=./:${LD_LIBRARY_PATH} ./deepspeech ${ARGS}

debug-libdeepspeech: client-libdeepspeech
	LD_LIBRARY_PATH=./:${LD_LIBRARY_PATH} gdb --args ./deepspeech ${ARGS}
