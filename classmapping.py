from pathlib import Path
import shutil
import zipfile
import os

from pathlib import Path
import shutil
import zipfile
import os

DRIVE = Path("/content/drive/MyDrive")

# 원본 데이터
TT_SPLIT = DRIVE / "TT100K_split"
GEN_ZIP = DRIVE / "Generated-TT100K-weather" / "Generated-TT100K-weather.zip"

# 새로 만들 3-class 복사본
TT_3CLS = DRIVE / "TT100K_split_3cls"

GEN_EXTRACT = Path("/content/Generated-TT100K-weather_45cls")
GEN_3CLS = Path("/content/Generated-TT100K-weather_3cls")
GEN_3CLS_ZIP = DRIVE / "Generated-TT100K-weather" / "Generated-TT100K-weather_3cls"


# 기존 45-class 순서
CLASS_NAMES_45 = [
    "i2", "i4", "i5", "il100", "il60", "il80", "io", "ip",
    "p10", "p11", "p12", "p19", "p23", "p26", "p27", "p3",
    "p5", "p6", "pg", "ph4", "ph4.5", "ph5",
    "pl100", "pl120", "pl20", "pl30", "pl40", "pl5", "pl50",
    "pl60", "pl70", "pl80", "pm20", "pm30", "pm55",
    "pn", "pne", "po", "pr40",
    "w13", "w32", "w55", "w57", "w59", "wo"
]

# CCTSDB2021과 동일
SUPERCLASS_NAMES = {
    0: "mandatory",
    1: "prohibitory",
    2: "warning"
}

def class_name_to_superclass(name):
    if name.startswith("i"):
        return 0
    elif name.startswith("p"):
        return 1
    elif name.startswith("w"):
        return 2
    else:
        raise ValueError(f"분류할 수 없는 클래스: {name}")

OLD_TO_NEW = {
    old_id: class_name_to_superclass(name)
    for old_id, name in enumerate(CLASS_NAMES_45)
}

for old_id, name in enumerate(CLASS_NAMES_45):
    new_id = OLD_TO_NEW[old_id]
    print(f"{old_id:2d} {name:6s} -> {new_id} {SUPERCLASS_NAMES[new_id]}") 

def convert_label_file(src, dst):
    """
    YOLO label:
    old_class x_center y_center width height
             ↓
    new_class x_center y_center width height
    """
    lines = src.read_text().splitlines()
    converted = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        old_id = int(parts[0])

        if old_id not in OLD_TO_NEW:
            raise ValueError(
                f"잘못된 class ID 발견: {old_id}\n파일: {src}"
            )

        new_id = OLD_TO_NEW[old_id]

        parts[0] = str(new_id)
        converted.append(" ".join(parts))

    dst.parent.mkdir(parents=True, exist_ok=True)

    dst.write_text(
        "\n".join(converted) + ("\n" if converted else "")
    )


def is_yolo_label(txt_path):
    try:
        lines = txt_path.read_text().splitlines()

        if len(lines) == 0:
            # 빈 annotation도 label 파일일 수 있음
            return True

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # YOLO detection label은 최소
            # class x y w h
            if len(parts) < 5:
                return False

            class_id = int(parts[0])

            if not (0 <= class_id < 45):
                return False

            # 나머지가 숫자인지 확인
            for x in parts[1:]:
                float(x)

        return True

    except:
        return False


def convert_dataset(src_root, dst_root):

    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if dst_root.exists():
        print(f"기존 출력 폴더 삭제: {dst_root}")
        shutil.rmtree(dst_root)

    file_count = 0
    label_count = 0

    for src in src_root.rglob("*"):

        if src.is_dir():
            continue

        relative = src.relative_to(src_root)
        dst = dst_root / relative

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.suffix.lower() == ".txt" and is_yolo_label(src):
            convert_label_file(src, dst)
            label_count += 1

        else:
            shutil.copy2(src, dst)

        file_count += 1

    print("=" * 50)
    print("변환 완료")
    print("전체 파일 :", file_count)
    print("변환 label:", label_count)
    print("저장 위치 :", dst_root)
  
