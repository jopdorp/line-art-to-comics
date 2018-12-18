#!/bin/bash

IMAGE_B=$1
PREFIX=$2
VGG_WEIGHTS=vgg16_weights.h5
HEIGHT=512
PATCH_SIZE=3  # try 3 for more interesting, but slow-rendering effects

SKULL_IMAGE_A=images/portrait-color_blur_2px.png
SKULL_IMAGE_AP=images/portrait-color.png

echo "Paint jom style (CPU)"
THEANO_FLAGS='openmp=1' OMP_NUM_THREADS=8 \
  make_image_analogy.py \
  $SKULL_IMAGE_A $SKULL_IMAGE_AP \
  $IMAGE_B \
  out/$PREFIX-sugarskull-cpu/$PREFIX-Bp  \
  --height=$HEIGHT \
  --b-content-w=1.5 \
  --mrf-w=1.5 \
  --a-scale-mode=match \
  --model=patchmatch --patch-size=$PATCH_SIZE \
  --contrast=1 \
  --vgg-weights=$VGG_WEIGHTS --output-full
