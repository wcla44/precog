#!/usr/bin/env bash
# Download the MediaPipe models and convert them to ONNX (one-off, needs `uv`).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/onnx build/tflite
base=https://storage.googleapis.com/mediapipe-models
[ -f models/face_landmarker.task ] || curl -fL -o models/face_landmarker.task $base/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
[ -f models/hand_landmarker.task ] || curl -fL -o models/hand_landmarker.task $base/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
unzip -o -q models/face_landmarker.task -d build/tflite
unzip -o -q models/hand_landmarker.task -d build/tflite
for m in face_detector face_landmarks_detector hand_detector hand_landmarks_detector; do
  uvx --python 3.12 --with tensorflow==2.18.0 --with onnx==1.17.0 --with "numpy<2" --from tf2onnx==1.16.1 \
    python -m tf2onnx.convert --tflite build/tflite/$m.tflite --output models/onnx/$m.onnx --opset 17
done
ls -la models/onnx
