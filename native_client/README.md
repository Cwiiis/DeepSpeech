# DeepSpeech native client

A native client for running queries on an exported DeepSpeech model.

## Requirements

* [Tensorflow](https://www.tensorflow.org/get_started/os_setup)
* [libsox](https://sourceforge.net/projects/sox/)

## Preparation

Create a symbolic link in the Tensorflow checkout to the deepspeech `native_client` directory.

```
cd tensorflow
ln -s ../DeepSpeech/native_client ./
```

## Building

To build the Tensorflow library, execute the following command:

```
bazel build -c opt //tensorflow:libtensorflow.so
```

Then you can build the DeepSpeech native library.

```
bazel build -c opt //native_client:deepspeech
```

Finally, you can change to the `native_client` directory and use the `Makefile`. By default, the `Makefile` will assume there is a TensorFlow checkout in a directory above the DeepSpeech checkout. If that is not the case, set the environment variable `TFDIR` to point to the right directory.

```
cd ../DeepSpeech/native_client
make deepspeech
```

## Running

The client can be run via the `Makefile`.

```
ARGS="/path/to/output_graph.pb /path/to/audio/file.ogg" make run
```
