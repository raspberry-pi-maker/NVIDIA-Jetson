#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Download TF Lite models
FILE=${DATA_DIR}/efficientdet_lite0.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite' \
    -o ${FILE}
fi
FILE=${DATA_DIR}/efficientdet_lite0_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/edgetpu/efficientdet_lite0_edgetpu_metadata.tflite' \
    -o ${FILE}
fi

FILE=${DATA_DIR}/efficientdet_lite1.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1?lite-format=tflite' \
    -o ${FILE}
fi
FILE=${DATA_DIR}/efficientdet_lite1_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/edgetpu/efficientdet_lite1_edgetpu_metadata.tflite' \
    -o ${FILE}
fi

FILE=${DATA_DIR}/efficientdet_lite2.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1?lite-format=tflite' \
    -o ${FILE}
fi
FILE=${DATA_DIR}/efficientdet_lite2_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/edgetpu/efficientdet_lite2_edgetpu_metadata.tflite' \
    -o ${FILE}
fi


FILE=${DATA_DIR}/efficientdet_lite3.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/tensorflow/lite-model/efficientdet/lite3/detection/metadata/1?lite-format=tflite' \
    -o ${FILE}
fi
FILE=${DATA_DIR}/efficientdet_lite3_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/edgetpu/efficientdet_lite3_edgetpu_metadata.tflite' \
    -o ${FILE}
fi

FILE=${DATA_DIR}/efficientdet_lite4.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/tensorflow/lite-model/efficientdet/lite4/detection/metadata/1?lite-format=tflite' \
    -o ${FILE}
fi
FILE=${DATA_DIR}/efficientdet_lite4_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/edgetpu/efficientdet_lite4_edgetpu_metadata.tflite' \
    -o ${FILE}
fi
echo -e "Downloaded files are in ${DATA_DIR}"