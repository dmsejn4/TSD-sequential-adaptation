# 환경 세팅
!nvidia-smi

from pathlib import Path

if not Path("/content/YOLO-TS").exists():
    !git clone https://github.com/Heqiang-Huang/YOLO-TS.git

!pip install -r /content/YOLO-TS/requirements.txt

# 드라이브 연결
from google.colab import drive
drive.mount('/content/drive')
%cd /content/YOLO-TS

# 데이터 경로 설정
from pathlib import Path

TT_TRAIN = Path("/content/drive/MyDrive/TT100K_split_3cls")
TT_TEST  = Path("/content/drive/MyDrive/TT100K_test_3cls")
GEN_ROOT = Path("/content/drive/MyDrive/GenTT100K_3cls")

print("TT train:", TT_TRAIN.exists())
print("TT test :", TT_TEST.exists())
print("GenTT   :", GEN_ROOT.exists())

# YOLO-TS의 yaml 파일 수정(nc:45 -> 3)
import re
from pathlib import Path

ROOT = Path("/content/YOLO-TS")

TT_MODEL_ORIG = ROOT / "YOLO-TS_TT100K.yaml"
TT_MODEL_3CLS = ROOT / "YOLO-TS_TT100K_3cls.yaml"

text = TT_MODEL_ORIG.read_text()

new_text, n = re.subn(
    r"^nc:\s*\d+",
    "nc: 3",
    text,
    count=1,
    flags=re.MULTILINE
)

assert n == 1, "TT model YAML의 nc 변경 실패"

TT_MODEL_3CLS.write_text(new_text)

print(TT_MODEL_3CLS)

ROOT = Path("/content/YOLO-TS")

TT_ROOT = Path("/content/drive/MyDrive/TT100K_split_3cls")
TT_TEST = Path("/content/drive/MyDrive/TT100K_test_3cls")

TT_DATA_YAML = ROOT / "TT100K_3cls.yaml"

tt_yaml = f"""
train: {TT_ROOT / 'train'}
val: {TT_ROOT / 'val'}
test: {TT_TEST}

nc: 3

names:
  0: mandatory
  1: prohibitory
  2: warning
"""

TT_DATA_YAML.write_text(tt_yaml.strip() + "\n")

print(TT_DATA_YAML.read_text())

# PyTorch 2.6+ 호환 우회
import torch
original_load = torch.load

def safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)

torch.load = safe_load

# 스크래치 학습(TT100K-3class)
from ultralytics import YOLO

# YAML 구조에서 새 모델 생성
model_A3 = YOLO(
    "/content/YOLO-TS/YOLO-TS_TT100K_3cls.yaml"
)

results_A3 = model_A3.train(
    data="/content/YOLO-TS/TT100K_3cls.yaml",

    epochs=200,
    batch=48,
    imgsz=640,

    device=0,
    pretrained=False,

    project="/content/drive/MyDrive/YOLO_TS_3cls_runs",
    name="Model_A3_TT100K",

    exist_ok=True
)
