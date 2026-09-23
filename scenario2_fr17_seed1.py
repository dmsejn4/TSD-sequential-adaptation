# 5. 전이학습 실행
results = model.train(
        data=data_path,
        imgsz=640,              # 비교 위해 고정
        epochs=150,
        patience=15,
        batch=16,
        nbs=64,

        # --- optimizer ---
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.01,
        cos_lr=True,
        warmup_epochs=5.0,

        # --- augmentation (명시적 지정) ---
        mosaic=1.0,
        close_mosaic=20,
        mixup=0.15,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.0,             # 표지판 방향성 보존
        flipud=0.0,

        # --- misc ---
        seed=1,
        deterministic=True,
        freeze=17,
        save_period=10,
        project='/content/drive/MyDrive/runs',
        name='scenario2_fr17_seed1',
        amp=True,
        workers=2,
        cache=False,  # 🔥 형태 보존을 위해 off
)

print("\n✅ TT100K 사전학습 모델 기반 전이학습 완료!")
from google.colab import runtime

runtime.unassign()
