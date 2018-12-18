#!/bin/bash

IMAGE_B=$1
PREFIX=$2
VGG_WEIGHTS=vgg16_weights.h5
HEIGHT=512
PATCH_SIZE=3  # try 3 for more interesting, but slow-rendering effects

SKULL_IMAGE_A=images/portrait-lines.png
SKULL_IMAGE_AP=images/portrait-color.png

echo "Paint jom style (CPU)"
  KERAS_BACKEND=theano \
  make_image_analogy.py \
  $SKULL_IMAGE_A $SKULL_IMAGE_AP \
  $IMAGE_B \
  out/$PREFIX-sugarskull-cpu/$PREFIX-Bp  \
  --height=$HEIGHT \
  --b-content-w=0 \
  --analogy-layers=conv2_1,conv3_1,conv4_1,conv5_1 \
  --mrf-layers=conv2_1,conv3_1 \
  --mrf-w=0.36 \
  --pool-mode=avg \
  --a-scale-mode=match \
  --model=patchmatch --patch-size=$PATCH_SIZE \
  --contrast=2 \
  --vgg-weights=$VGG_WEIGHTS --output-full
