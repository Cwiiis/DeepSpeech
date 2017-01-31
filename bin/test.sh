
#!/bin/sh

set -xe

do_inference () {
  echo '--- $1, dropout = $2, $3, RNN dropout = $4, $5, $6, $7, swapout = $8, $9'

  export ds_importer=$1

  export ds_dropout_rate0=$2
  export ds_dropout_rate1=`echo "(($3-$2)/3.0)+$2"|bc -l`
  export ds_dropout_rate2=`echo "(($3-$2)/3.0*2.0)+$2"|bc -l`
  export ds_dropout_rate7=$3

  export ds_dropout_rate3=$4
  export ds_dropout_rate4=$5
  export ds_dropout_rate5=$6
  export ds_dropout_rate6=$7

  export ds_swapout_rate0=$8
  export ds_swapout_rate1=`echo "(($9-$8)/3.0)+$8"|bc -l`
  export ds_swapout_rate2=`echo "(($9-$8)/3.0*2.0)+$8"|bc -l`
  export ds_swapout_rate3=$9

  jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python -u
}

export ds_train_batch_size=1
export ds_dev_batch_size=1
export ds_test_batch_size=1
export ds_epochs=200
export ds_validation_step=0

# Testing with LDC to see how variables affect convergence speed
# Test feed-forward layers with varying levels of dropout
do_inference ldc93s1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
do_inference ldc93s1 0.1 0.1 0.0 0.0 0.0 0.0 0.0 0.0
do_inference ldc93s1 0.2 0.2 0.0 0.0 0.0 0.0 0.0 0.0
do_inference ldc93s1 0.3 0.3 0.0 0.0 0.0 0.0 0.0 0.0
do_inference ldc93s1 0.4 0.4 0.0 0.0 0.0 0.0 0.0 0.0

# Same thing with linearly increasing dropout
do_inference ldc93s1 0.1 0.4 0.0 0.0 0.0 0.0 0.0 0.0

# Same thing with ResNet + swapout
do_inference ldc93s1 0.1 0.1 0.0 0.0 0.0 0.0 0.1 0.1
do_inference ldc93s1 0.2 0.2 0.0 0.0 0.0 0.0 0.2 0.2
do_inference ldc93s1 0.3 0.3 0.0 0.0 0.0 0.0 0.3 0.3
do_inference ldc93s1 0.4 0.4 0.0 0.0 0.0 0.0 0.4 0.4

# Same thing with ResNet + linearly increasing swapout
do_inference ldc93s1 0.1 0.4 0.0 0.0 0.0 0.0 0.1 0.4

# With TED

export ds_train_batch_size=16
export ds_dev_batch_size=8
export ds_test_batch_size=8
export ds_learning_rate=0.0001
export ds_validation_step=20
export ds_display_step=10
export ds_checkpoint_step=10
export ds_limit_train=500
export ds_limit_dev=500
export ds_limit_test=500

# First with no ResNet and 0.3 dropout
do_inference ted 0.3 0.3 0.0 0.0 0.0 0.0 0.0 0.0
# Then with linearly increasing dropout
do_inference ted 0.1 0.4 0.0 0.0 0.0 0.0 0.0 0.0
# Then with linearly increasing dropout + RNN dropout
do_inference ted 0.1 0.4 0.2 0.2 0.2 0.2 0.0 0.0
# Then with ResNet and 0.3 swapout
do_inference ted 0.3 0.3 0.0 0.0 0.0 0.0 0.3 0.3
# Then with ResNet and linearly increasing swapout
do_inference ted 0.1 0.4 0.0 0.0 0.0 0.0 0.1 0.4
# Then with ResNet and linearly increasing swapout + RNN dropout
do_inference ted 0.1 0.4 0.2 0.2 0.2 0.2 0.1 0.4
