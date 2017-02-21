
TFDIR ?= ../../tensorflow

default: deepspeech

clean:
	rm -f deepspeech

deepspeech: client.cc
	c++ -o deepspeech client.cc `pkg-config --cflags --libs sox` -L${TFDIR}/bazel-bin/tensorflow -L${TFDIR}/bazel-bin/native_client -ldeepspeech -ltensorflow

run: deepspeech
	LD_LIBRARY_PATH=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${LD_LIBRARY_PATH} ./deepspeech ${ARGS}
