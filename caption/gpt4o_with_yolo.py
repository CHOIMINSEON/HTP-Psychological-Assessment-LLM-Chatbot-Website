"""
GPT-4o를 사용한 이미지 캡셔닝 (YOLO 탐지 결과와 결합)
1. GPT-4o 단독 캡셔닝
2. YOLOv8 객체 탐지
3. GPT-4o + YOLO 결합 캡셔닝
필요한 패키지: openai, pillow, ultralytics, google-colab (Colab 환경)
"""

import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from ultralytics import YOLO


def get_openai_client():
    """OpenAI 클라이언트 초기화 (Colab 보안 비밀 사용)"""
    try:
        # Colab 환경인 경우
        from google.colab import userdata
        api_key = userdata.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Colab 보안 비밀에서 'OPENAI_API_KEY'를 찾을 수 없음.")
    except ImportError:
        # 로컬 환경인 경우
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("환경 변수에서 'OPENAI_API_KEY'를 찾을 수 없음.")
    
    client = OpenAI(api_key=api_key)
    print("✅ OpenAI 클라이언트 초기화 완료.")
    return client


def encode_image_to_base64(image_path):
    """로컬 이미지 파일을 Base64 문자열로 변환"""
    try:
        img = Image.open(image_path).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except FileNotFoundError:
        print(f"❌ 오류: 이미지 경로를 찾을 수 없음: {image_path}")
        return None
    except Exception as e:
        print(f"❌ 오류: 이미지 변환 중 문제 발생: {e}")
        return None


def generate_caption_gpt4o(client, image_base64, model_name="gpt-4o"):
    """GPT-4o로 캡션을 생성"""
    if not image_base64:
        return "이미지 인코딩 실패로 캡션 생성 실패."
    
    print(f"⏳ GPT 모델({model_name})에 캡션 요청 중...")
    
    caption_prompt = "이 이미지를 자세하고 간결하게 설명해줘. 이미지에 보이는 주요 물체와 장면의 분위기를 포함해."
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        
        caption = response.choices[0].message.content
        return caption.strip()
    
    except Exception as e:
        return f"❌ API 호출 오류 발생: {e}"


def get_yolo_detections(model_path, image_path):
    """YOLOv8 모델로 객체를 탐지하고 결과를 텍스트로 정리"""
    print("⏳ YOLO 모델 로딩 및 탐지 시작...")
    try:
        model = YOLO(model_path)
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
        
        detection_string = "탐지된 객체 목록: " + ", ".join(detections)
        print("✅ YOLO 탐지 완료.")
        return detection_string
    
    except Exception as e:
        return f"❌ YOLO 탐지 오류: {e}"


def generate_caption_with_yolo(client, image_base64, yolo_detections, model_name="gpt-4o"):
    """YOLO 탐지 정보를 활용하여 GPT-4o로 캡션 생성"""
    caption_prompt = (
        "이 이미지를 자세하고 간결하게 설명해줘. 다음 YOLOv8 탐지 결과를 참고해서 이미지 내용을 더 정확하게 묘사해줘. "
        f"\n\n[YOLO 탐지 정보]: {yolo_detections}"
    )
    
    print(f"⏳ GPT 모델({model_name})에 캡션 요청 중 (YOLO 정보 포함)...")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        
        caption = response.choices[0].message.content
        return caption.strip()
    
    except Exception as e:
        return f"❌ API 호출 오류 발생: {e}"


def main(image_path, yolo_model_path=None):
    """메인 실행 함수"""
    # OpenAI 클라이언트 초기화
    try:
        client = get_openai_client()
    except Exception as e:
        print(f"❌ 오류: {e}")
        return
    
    # 이미지 Base64 인코딩
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return
    
    # GPT-4o 단독 캡션
    print("\n" + "=" * 50)
    print("📸 GPT-4o Caption (단독)")
    print("=" * 50)
    gpt4o_caption = generate_caption_gpt4o(client, base64_image)
    print(gpt4o_caption)
    
    # YOLO를 사용하는 경우
    if yolo_model_path:
        print("\n" + "=" * 50)
        print("🔍 YOLOv8 탐지 결과")
        print("=" * 50)
        yolo_output = get_yolo_detections(yolo_model_path, image_path)
        print(yolo_output)
        
        print("\n" + "=" * 50)
        print("📸 GPT-4o Caption (YOLO 결합)")
        print("=" * 50)
        combined_caption = generate_caption_with_yolo(client, base64_image, yolo_output)
        print(combined_caption)
        
        return {
            "gpt4o_only": gpt4o_caption,
            "yolo_detection": yolo_output,
            "gpt4o_with_yolo": combined_caption
        }
    
    return {"gpt4o_only": gpt4o_caption}


if __name__ == "__main__":
    # 이미지 경로 설정 (사용자가 수정해야 함)
    IMAGE_PATH = "/content/drive/MyDrive/Colab/T_V_T/htp/test_나무.JPG"
    YOLO_MODEL_PATH = "/content/drive/MyDrive/Colab/T_V_T/pt/68_100best.pt"  # 선택사항
    
    # YOLO 모델 경로가 있으면 결합 버전 실행, 없으면 GPT-4o만 실행
    main(IMAGE_PATH, YOLO_MODEL_PATH)
