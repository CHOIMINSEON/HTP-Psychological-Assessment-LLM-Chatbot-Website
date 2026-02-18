"""
InstructBLIP + YOLOv8 결합 캡셔닝
YOLOv8 탐지 결과를 프롬프트에 포함하여 InstructBLIP로 상세한 캡션 생성
필요한 패키지: transformers, accelerate, torch, torchvision, ultralytics, pillow
"""

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import torch
from ultralytics import YOLO
import os


def setup_device():
    """GPU 사용 가능 여부 확인"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")
    return device


def get_yolo_detections(model_path, image_path):
    """YOLOv8 모델로 객체를 탐지하고 결과를 텍스트로 정리"""
    print("⏳ YOLO 모델 로딩 및 탐지 시작...")
    try:
        model = YOLO(model_path)
        # 이미지 탐지 실행 (확신도 0.5 이상, IOU 0.7 이상)
        results = model(image_path, conf=0.5, iou=0.7, save=False, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = model.names.get(class_id, f"class_{class_id}")
                detections.append(f"{class_name} (확률: {confidence:.2f})")
        
        if not detections:
            return "탐지된 객체 없음."
        
        detection_string = "YOLO 탐지 객체: " + ", ".join(detections)
        return detection_string
    
    except Exception as e:
        return f"❌ YOLO 탐지 오류: {e}"


def generate_instructblip_caption_with_yolo(image, yolo_output, device):
    """InstructBLIP으로 YOLO 결과를 포함한 캡션 생성"""
    print("🔹 Running InstructBLIP with YOLO Hint...")
    
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    # 메모리 절약을 위해 float16을 사용하여 모델을 로드
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl", 
        torch_dtype=torch.float16
    ).to(device)
    
    # InstructBLIP 프롬프트에 YOLO 결과 추가
    base_prompt = "Describe this image in detail, including objects, colors, positions, sizes, shapes, and atmosphere."
    yolo_hint = f"참고 정보 (YOLO 탐지 결과): {yolo_output}. 이 정보를 바탕으로 이미지 설명을 더 정확하게 작성해줘."
    final_prompt = f"{base_prompt} {yolo_hint}"
    
    print(f"📝 최종 프롬프트: {final_prompt}")
    
    # 모델 입력 준비
    inputs = processor(images=image, text=final_prompt, return_tensors="pt").to(device)
    
    # 캡션 생성 (충분히 자세한 설명을 위해 max_new_tokens 설정)
    out = model.generate(**inputs, max_new_tokens=150)
    caption = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    
    return caption


def main(image_path, yolo_model_path):
    """메인 실행 함수"""
    # 디바이스 설정
    device = setup_device()
    
    # 이미지 불러오기
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"✅ 이미지 불러오기 완료: {image_path}")
    except FileNotFoundError:
        print(f"❌ 오류: 이미지 경로를 찾을 수 없음: {image_path}")
        return
    
    # YOLO 탐지 실행
    yolo_output = get_yolo_detections(yolo_model_path, image_path)
    print(f"\n--- YOLO 탐지 결과 ---\n{yolo_output}\n" + "-" * 50)
    
    # InstructBLIP + YOLO 캡션 생성
    caption = generate_instructblip_caption_with_yolo(image, yolo_output, device)
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("📸 InstructBLIP + YOLO 캡션")
    print("=" * 50)
    print(f"🔸 InstructBLIP:")
    print(caption)
    
    return {
        "yolo_detection": yolo_output,
        "instructblip_caption": caption
    }


if __name__ == "__main__":
    # 이미지 경로 설정 (사용자가 수정해야 함)
    IMAGE_PATH = "/content/drive/MyDrive/Colab/T_V_T/htp/test_나무.JPG"
    YOLO_MODEL_PATH = "/content/drive/MyDrive/Colab/T_V_T/pt/68_100best.pt"
    
    main(IMAGE_PATH, YOLO_MODEL_PATH)
