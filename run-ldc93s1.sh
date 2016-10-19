#!/bin/sh

set -xe

ds_dataset_path="./data/ldc93s1/"
export ds_dataset_path

ds_importer="ldc93s1"
export ds_importer

ds_batch_size=2
export ds_batch_size

jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python
