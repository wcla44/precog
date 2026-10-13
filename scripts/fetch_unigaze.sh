#!/usr/bin/env bash
# Fetch UniGaze-B (ut-vision/UniGaze), export it to ONNX and quantize it to int8 -> models/onnx/unigaze_b16_int8.onnx
# The weights are under a non-commercial license (MG-NC-RAI-2.0): they are never committed (models/ is gitignored).
# Needs uv and ~2 GB of temporary disk (torch). The temporary environment is deleted at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=models/onnx/unigaze_b16_int8.onnx
[ -f "$OUT" ] && { echo "$OUT already exists"; exit 0; }
W=build/unigaze
mkdir -p "$W" models/onnx
[ -d "$W/repo" ] || git clone -q https://github.com/ut-vision/UniGaze "$W/repo"
git -C "$W/repo" checkout -q 9c240fb
[ -f "$W/unigaze_b16_joint.safetensors" ] || curl -fL -o "$W/unigaze_b16_joint.safetensors" \
  https://huggingface.co/UniGaze/UniGaze-models/resolve/main/unigaze_b16_joint.safetensors
uv venv -q --python 3.12 "$W/venv"
VIRTUAL_ENV="$W/venv" uv pip install -q torch==2.14.1 torchvision timm==1.0.30 safetensors onnx onnxruntime==1.31.0 numpy
"$W/venv/bin/python" -I - "$W/repo/unigaze_easy/src" "$W/unigaze_b16_joint.safetensors" "$W/unigaze_b16.onnx" "$OUT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from unigaze.models.mae_gaze import MAE_Gaze

m = MAE_Gaze(model_type="vit_b_16")
m.load_unigaze_weights(sys.argv[2])
m.eval()

class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):  # x: (n, 3, 224, 224), ImageNet-normalised RGB of the ETH-XGaze-normalised face
        return self.m(x)["pred_gaze"]  # (n, 2) pitch, yaw in radians, normalised camera space

torch.onnx.export(Wrap(m), torch.randn(1, 3, 224, 224), sys.argv[3], input_names=["image"], output_names=["pitchyaw"],
                  opset_version=17, dynamo=False, dynamic_axes={"image": {0: "n"}})
quantize_dynamic(sys.argv[3], sys.argv[4], weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
print("wrote", sys.argv[4])
PY
rm -rf "$W/venv" "$W/unigaze_b16.onnx"
ls -la "$OUT"
