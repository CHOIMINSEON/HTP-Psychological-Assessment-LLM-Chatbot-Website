"""
YOLOv8 및 Detectron2 모델 학습
- YOLOv8 학습 (속도 최적화)
- Detectron2 학습 (Early Stopping 포함)
- 드라이브 → 로컬 복사 옵션 (I/O 개선)
필요한 패키지: ultralytics, detectron2, torch, yaml
"""

import os
import yaml
import torch
from ultralytics import YOLO
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog


# ===============================
# 설정 변수
# ===============================
# Drive → 로컬 복사 설정 (Colab 환경에서 I/O 개선)
COPY_TO_LOCAL = True
DATA_ROOT_DRIVE = r"/content/drive/MyDrive/Colab/T_V_T"
DATA_ROOT_LOCAL = r"/content/dataset/T_V_T"

# 클래스 이름 정의
CLASS_NAMES = [
    '집전체','지붕','집벽','문','창문','굴뚝','연기','울타리','길','연못','산','나무','꽃','잔디','태양',
    '나무전체','기둥','수관','가지','뿌리','나뭇잎','꽃','열매','그네','새','다람쥐','구름','달','별',
    '사람전체','머리','얼굴','눈','코','입','귀','머리카락','목','상체','팔','손','다리','발','단추','주머니','운동화','여자구두',
    '사람전체','머리','얼굴','눈','코','입','귀','머리카락','목','상체','팔','손','다리','발','단추','주머니','운동화','남자구두'
]


def copy_data_to_local():
    """Drive → 로컬 복사 (I/O 개선)"""
    if COPY_TO_LOCAL:
        if not os.path.exists(DATA_ROOT_LOCAL):
            print("복사 시작: Drive → 로컬 (시간이 걸릴 수 있지만 I/O를 크게 개선합니다)...")
            os.system(f"cp -r '{DATA_ROOT_DRIVE}' '/content/dataset/'")
            
            if os.path.exists(DATA_ROOT_LOCAL):
                print("✅ 복사 완료:", DATA_ROOT_LOCAL)
                return DATA_ROOT_LOCAL
            else:
                print("⚠️ 복사 실패: Drive 경로를 사용합니다.")
                return DATA_ROOT_DRIVE
        else:
            print("✅ 로컬 데이터 이미 존재:", DATA_ROOT_LOCAL)
            return DATA_ROOT_LOCAL
    else:
        return DATA_ROOT_DRIVE


def create_yolo_yaml(data_root):
    """YOLOv8용 데이터 YAML 생성"""
    yolo_data_yaml = os.path.join(data_root, "yolo_data.yaml")
    
    yolo_data = {
        'train': os.path.join(data_root, "train/images"),
        'val': os.path.join(data_root, "val/images"),
        'nc': 65,
        'names': CLASS_NAMES
    }
    
    with open(yolo_data_yaml, 'w') as f:
        yaml.dump(yolo_data, f)
    
    print(f"✅ YOLO YAML 생성 완료: {yolo_data_yaml}")
    return yolo_data_yaml


def train_yolo(yolo_data_yaml, data_root, resume=False, last_model_path=None):
    """YOLOv8 학습"""
    print("\n" + "=" * 50)
    print("YOLOv8 학습 시작")
    print("=" * 50)
    
    # 모델 로드
    if resume and last_model_path and os.path.exists(last_model_path):
        print(f"✅ 이전 모델 불러오기: {last_model_path}")
        model = YOLO(last_model_path)
    else:
        print("✅ 새 모델 시작: yolov8s.pt")
        model = YOLO('yolov8s.pt')
    
    yolo_project = os.path.join(data_root, "YOLO_train")
    os.makedirs(yolo_project, exist_ok=True)
    
    # 학습 실행
    model.train(
        data=yolo_data_yaml,
        epochs=100,
        patience=5,
        batch=64,
        imgsz=640,
        resume=resume,
        project=yolo_project,
        name="yolo_TV_T_fast",
        exist_ok=True,
        cache='disk',
        workers=16,
        val_period=5,
        device=0,
        plots=True,
        save=True
    )
    
    print("✅ YOLOv8 학습 완료")


class EarlyStoppingTrainer(DefaultTrainer):
    """Early Stopping을 지원하는 Detectron2 Trainer"""
    
    def __init__(self, cfg, patience=5):
        super().__init__(cfg)
        self.best_metric = 0.0
        self.patience = patience
        self.counter = 0
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    def after_step(self):
        super().after_step()
        iteration = self.iter + 1
        
        # validation 빈도 조정
        if iteration % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == self.cfg.SOLVER.MAX_ITER:
            evaluator = self.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0])
            val_results = self.test(self.cfg, self.model, evaluators=[evaluator])
            
            # mAP50 읽기
            try:
                map_50 = val_results["bbox"]["AP50"]
            except Exception:
                map_50 = val_results.get("bbox", {}).get("AP50", 0.0)
            
            if map_50 > self.best_metric:
                self.best_metric = map_50
                self.counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, "best_model.pth"))
                print(f"✅ [Iteration {iteration}] mAP50 개선: {map_50:.4f}")
            else:
                self.counter += 1
                print(f"⚠️ [Iteration {iteration}] 개선 없음 ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                print("🛑 얼리 스탑핑 발동 - 학습 중단")
                raise SystemExit


def train_detectron2(data_root):
    """Detectron2 학습"""
    print("\n" + "=" * 50)
    print("Detectron2 학습 시작")
    print("=" * 50)
    
    # 기존 등록 데이터셋 제거
    for d in ["TVT_train", "TVT_val"]:
        if d in DatasetCatalog.list():
            DatasetCatalog.remove(d)
        if d in MetadataCatalog.list():
            MetadataCatalog.remove(d)
    
    # COCO 데이터셋 등록
    coco_train_json = os.path.join(data_root, "train/labels/Detectron2/coco_train.json")
    coco_val_json = os.path.join(data_root, "val/labels/Detectron2/coco_val.json")
    
    register_coco_instances("TVT_train", {}, coco_train_json, os.path.join(data_root, "train/images"))
    register_coco_instances("TVT_val", {}, coco_val_json, os.path.join(data_root, "val/images"))
    
    # Config 설정
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # 데이터셋
    cfg.DATASETS.TRAIN = ("TVT_train",)
    cfg.DATASETS.TEST = ("TVT_val",)
    
    # 데이터 로딩
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    # 이미지 리사이징
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1280
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 1280
    
    # 모델 설정
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 65
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    
    # 학습 하이퍼파라미터
    cfg.SOLVER.IMS_PER_BATCH = 32
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 2500
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    
    # 출력 경로
    cfg.OUTPUT_DIR = os.path.join(data_root, "Detectron2_train")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 학습 실행
    trainer = EarlyStoppingTrainer(cfg, patience=5)
    trainer.resume_or_load(resume=False)
    
    try:
        trainer.train()
        print("✅ Detectron2 학습 완료")
    except SystemExit:
        print("✅ Detectron2 학습 완료 (Early Stopping)")


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("모델 학습 시작")
    print("=" * 50)
    
    # 1. 데이터 경로 설정 (로컬 복사 옵션)
    data_root = copy_data_to_local()
    print(f"데이터 경로: {data_root}\n")
    
    # 2. YOLO YAML 생성
    yolo_data_yaml = create_yolo_yaml(data_root)
    
    # 3. YOLOv8 학습
    train_yolo(yolo_data_yaml, data_root, resume=False)
    
    # 4. Detectron2 학습
    train_detectron2(data_root)
    
    print("\n" + "=" * 50)
    print("모든 학습 완료")
    print("=" * 50)


if __name__ == "__main__":
    main()
