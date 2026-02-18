"""
BLIP, InstructBLIP, CLIP Interrogator를 사용한 이미지 캡셔닝 비교
필요한 패키지: transformers, pillow, accelerate, torch, torchvision, clip-interrogator
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import torch


def setup_device():
    """GPU 사용 가능 여부 확인"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")
    return device


def generate_blip_caption(image, device):
    """BLIP 모델로 캡션 생성"""
    print("🔹 Running BLIP...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=100)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption


def generate_instructblip_caption(image, device, prompt=None):
    """InstructBLIP 모델로 캡션 생성"""
    print("🔹 Running InstructBLIP...")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").to(device)
    
    if prompt is None:
        prompt = "Describe this image in detail, including objects, colors, positions, sizes, shapes, and atmosphere."
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=100)
    caption = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    
    return caption


def generate_clip_interrogator_caption(image):
    """CLIP Interrogator로 캡션 생성"""
    try:
        print("🔹 Running CLIP Interrogator...")
        from clip_interrogator import Config, Interrogator
        import open_clip
        
        ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        caption = ci.interrogate(image)
        return caption
    except Exception as e:
        return f"⚠️ CLIP Interrogator not run: {e}"


def main(image_path):
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
    
    # 결과 저장용 딕셔너리
    results = {}
    
    # BLIP 캡션
    results["BLIP"] = generate_blip_caption(image, device)
    
    # InstructBLIP 캡션
    results["InstructBLIP"] = generate_instructblip_caption(image, device)
    
    # CLIP Interrogator 캡션
    results["CLIP Interrogator"] = generate_clip_interrogator_caption(image)
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("📸 Detailed Caption Comparison")
    print("=" * 50)
    for model_name, caption in results.items():
        print(f"\n🔸 {model_name}:")
        print(caption)
    
    return results


if __name__ == "__main__":
    # 이미지 경로 설정 (사용자가 수정해야 함)
    IMAGE_PATH = "/content/drive/MyDrive/Colab/T_V_T/htp/test_나무.JPG"
    
    main(IMAGE_PATH)
