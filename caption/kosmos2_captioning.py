"""
Kosmos-2 모델을 사용한 이미지 캡셔닝 (3가지 접근 방법)
1. 기본 출력
2. 노이즈 제거 버전
3. 프롬프트 사용 및 강화된 정리
필요한 패키지: transformers, accelerate, pillow, torchvision
"""

from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import re


def load_kosmos2_model(device):
    """Kosmos-2 모델 로드"""
    print("⏳ Kosmos-2 모델 로딩 시작...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    model = AutoModelForVision2Seq.from_pretrained(
        "microsoft/kosmos-2-patch14-224", 
        torch_dtype=torch.float16
    )
    model.to(device)
    print("✅ 모델 로딩 완료.")
    return processor, model


def generate_caption_basic(image, processor, model, device):
    """기본 캡션 생성 (최소 처리)"""
    prompt = ""
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    # 최소한의 정리: 이미지 관련 특수 토큰만 제거
    caption = caption.replace("<image>", "").replace("</image>", "").strip()
    
    return caption


def generate_caption_clean(image, processor, model, device):
    """노이즈 제거 버전"""
    prompt = ""
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    # 🌟 노이즈 및 불필요 문자열 제거
    # 1. 특수 토큰 제거
    caption = caption.replace("<image>", "").replace("</image>", "").replace("<grounding>", "").strip()
    
    # 2. HTML/XML 태그 형태 제거
    caption = re.sub(r'<[^>]+>', '', caption).strip()
    
    # 3. 대문자로 시작하는 부분 찾아서 그 앞 제거
    match = re.search(r'[A-Z]', caption)
    if match:
        caption = caption[match.start():].strip()
    else:
        caption = re.sub(r'^\s*[\.,:;!]+\s*', '', caption).strip()
    
    # 4. 소문자 시작 단어 정리
    caption = re.sub(r'^(the|to|and|of|as|in|I|that|for|is|was|on|it)\s*', '', caption, flags=re.IGNORECASE).strip()
    
    return caption


def generate_caption_with_prompt(image, processor, model, device):
    """프롬프트 사용 및 강화된 노이즈 제거"""
    # 명확한 프롬프트 사용
    prompt = "<grounding>A detailed description of the image, including all visible objects and their attributes:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=150)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    # 🌟 강화된 노이즈 및 불필요 문자열 제거
    # 1. 특수 토큰 제거
    caption = caption.replace("<image>", "").replace("</image>", "").replace("<grounding>", "").strip()
    
    # 2. HTML/XML 태그 형태 제거
    caption = re.sub(r'<[^>]+>', '', caption).strip()
    
    # 3. 대문자로 시작하는 부분 찾기
    match = re.search(r'[A-Z]', caption)
    if match:
        caption = caption[match.start():].strip()
    else:
        caption = re.sub(r'^\s*[\.,:;!]+\s*', '', caption).strip()
    
    # 4. 프롬프트가 캡션에 포함될 경우 제거
    caption = re.sub(re.escape(prompt.replace("<grounding>", "")), '', caption, flags=re.IGNORECASE, count=1).strip()
    
    # 5. 소문자 시작 단어 정리
    caption = re.sub(r'^(the|to|and|of|as|in|I|that|for|is|was|on|it)\s*', '', caption, flags=re.IGNORECASE).strip()
    
    # 6. 문장 끝 정리
    last_punc_match = re.search(r'[.?!](?=[^.?!]*$)', caption)
    if last_punc_match:
        caption = caption[:last_punc_match.end()].strip()
    
    return caption


def main(image_path):
    """메인 실행 함수"""
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")
    
    # 이미지 불러오기
    try:
        image = Image.open(image_path).convert("RGB")
        print("✅ 이미지 불러오기 완료.")
    except FileNotFoundError:
        print(f"❌ 오류: 이미지 경로를 찾을 수 없음: {image_path}")
        return
    
    # 모델 로드
    processor, model = load_kosmos2_model(device)
    
    # 3가지 방법으로 캡션 생성
    print("\n⏳ 캡션 생성 중...")
    
    caption_basic = generate_caption_basic(image, processor, model, device)
    caption_clean = generate_caption_clean(image, processor, model, device)
    caption_prompt = generate_caption_with_prompt(image, processor, model, device)
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("📸 Kosmos-2 Caption Comparison")
    print("=" * 50)
    print("\n🔸 Kosmos-2 Caption (기본 출력):")
    print(caption_basic)
    print("\n🔸 Kosmos-2 Caption (노이즈 제거 후):")
    print(caption_clean)
    print("\n🔸 Kosmos-2 Caption (프롬프트 사용 및 강화된 정리):")
    print(caption_prompt)
    
    return {
        "basic": caption_basic,
        "clean": caption_clean,
        "prompt": caption_prompt
    }


if __name__ == "__main__":
    # 이미지 경로 설정 (사용자가 수정해야 함)
    IMAGE_PATH = "/content/drive/MyDrive/Colab/T_V_T/htp/test_나무.JPG"
    
    main(IMAGE_PATH)
