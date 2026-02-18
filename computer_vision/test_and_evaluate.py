"""
YOLOv8 모델 테스트 및 평가
- 테스트 이미지 예측
- 성능 지표 계산 (mAP, precision, recall)
- 결과 CSV 저장
필요한 패키지: ultralytics, yaml, csv, tqdm
"""

import os
import glob
import yaml
import csv
from ultralytics import YOLO
from tqdm import tqdm


CLASS_NAMES = [
    '집전체','지붕','집벽','문','창문','굴뚝','연기','울타리','길','연못','산','나무','꽃','잔디','태양',
    '나무전체','기둥','수관','가지','뿌리','나뭇잎','꽃','열매','그네','새','다람쥐','구름','달','별',
    '사람전체','머리','얼굴','눈','코','입','귀','머리카락','목','상체','팔','손','다리','발','단추','주머니','운동화','여자구두',
    '사람전체','머리','얼굴','눈','코','입','귀','머리카락','목','상체','팔','손','다리','발','단추','주머니','운동화','남자구두'
]


def predict_batch(model, test_images, batch_size=32):
    """배치 단위로 예측 수행 (진행률 표시)"""
    print(f"⏳ 예측 시작 (총 {len(test_images)}개 이미지, 배치 크기: {batch_size})")
    
    results_all = []
    
    for i in tqdm(range(0, len(test_images), batch_size), desc="Predicting"):
        batch_imgs = test_images[i:i+batch_size]
        
        results = model.predict(
            source=batch_imgs,
            imgsz=640,
            conf=0.25,
            save=False,
            save_txt=False,
            verbose=False
        )
        results_all.extend(results)
    
    print("✅ 예측 완료")
    return results_all


def save_results_to_csv(results_all, save_path):
    """예측 결과를 CSV 파일로 저장"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # CSV 헤더
        writer.writerow(["image", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        
        for result in results_all:
            image_path = result.path
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                writer.writerow([image_path, "None", "None", "", "", "", ""])
                continue
            
            for box in boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                writer.writerow([image_path, cls, f"{conf:.4f}", x1, y1, x2, y2])
    
    print(f"✅ CSV 저장 완료: {save_path}")


def create_test_yaml(test_root, output_path):
    """테스트용 YAML 생성"""
    test_data = {
        'train': "",
        'val': os.path.join(test_root, "images"),
        'nc': 65,
        'names': CLASS_NAMES
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(test_data, f, allow_unicode=True)
    
    print(f"✅ 테스트 YAML 생성 완료: {output_path}")


def validate_model(model, test_yaml):
    """모델 검증 (mAP, Precision, Recall 계산)"""
    print("⏳ 검증 시작...")
    
    metrics = model.val(
        data=test_yaml,
        imgsz=640,
        batch=32,
        save_json=True,
        workers=4
    )
    
    print("✅ 검증 완료")
    return metrics


def test_with_prediction_and_save(model_path, test_root, output_dir, batch_size=32):
    """예측 수행 및 결과 저장"""
    print("\n" + "=" * 50)
    print("테스트 (예측) 모드")
    print("=" * 50)
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 테스트 이미지 불러오기
    test_images = glob.glob(os.path.join(test_root, "images", "**", "*.*"), recursive=True)
    print(f"✅ 총 {len(test_images)}개 이미지 확인됨")
    
    if not test_images:
        print("⚠️ 테스트 이미지가 없습니다.")
        return None
    
    # 배치 예측
    results_all = predict_batch(model, test_images, batch_size)
    
    # CSV 저장
    csv_path = os.path.join(output_dir, "test_results.csv")
    save_results_to_csv(results_all, csv_path)
    
    return results_all


def test_with_validation(model_path, test_root, output_dir):
    """검증 모드 (mAP, Precision, Recall 계산)"""
    print("\n" + "=" * 50)
    print("테스트 (검증) 모드")
    print("=" * 50)
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 테스트 YAML 생성
    test_yaml = os.path.join(test_root, "yolo_test.yaml")
    create_test_yaml(test_root, test_yaml)
    
    # 검증 실행
    metrics = validate_model(model, test_yaml)
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("검증 결과")
    print("=" * 50)
    print(metrics)
    
    return metrics


def main(model_path, test_root, output_dir, mode="both", batch_size=32):
    """
    메인 실행 함수
    
    Args:
        model_path: 학습된 모델 경로 (.pt 파일)
        test_root: 테스트 데이터 루트 경로
        output_dir: 결과 저장 경로
        mode: "predict" (예측만), "validate" (검증만), "both" (둘 다)
        batch_size: 배치 크기
    """
    print("=" * 50)
    print("YOLOv8 모델 테스트 시작")
    print("=" * 50)
    print(f"모델: {model_path}")
    print(f"테스트 데이터: {test_root}")
    print(f"출력 경로: {output_dir}")
    print(f"모드: {mode}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 예측 수행
    if mode in ["predict", "both"]:
        results["predictions"] = test_with_prediction_and_save(
            model_path, test_root, output_dir, batch_size
        )
    
    # 검증 수행
    if mode in ["validate", "both"]:
        results["metrics"] = test_with_validation(
            model_path, test_root, output_dir
        )
    
    print("\n" + "=" * 50)
    print("테스트 완료")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    # 설정 (사용자가 수정해야 함)
    MODEL_PATH = "/content/drive/MyDrive/Colab/T_V_T/YOLO_train_100/yolo_TV_T_fast/weights/best.pt"
    TEST_ROOT = "/content/drive/MyDrive/Colab/T_V_T/test"
    OUTPUT_DIR = "/content/drive/MyDrive/Colab/T_V_T/output/test_results"
    
    # 실행
    # mode: "predict" (예측만), "validate" (검증만), "both" (둘 다)
    main(MODEL_PATH, TEST_ROOT, OUTPUT_DIR, mode="both", batch_size=32)
