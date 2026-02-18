# 심리검사 LLM 챗봇 웹사이트(2025)
* 집-나무-사람(HTP) 그림 검사를 온라인으로 진행하는 웹사이트 제작.
* LLM 모델을 파인튜닝하여 HTP 심리검사 해석을 진행.
* GPT api를 활용해 채팅을 통해 상담 진행.
---
## 📂 Directory Structure

```text
HTP-Psychological-Assessment-LLM-Chatbot-Website/
│
├── 📂 caption/                      # 이미지 캡셔닝 모듈
│   ├── blip_clip_models.py          # BLIP, InstructBLIP, CLIP Interrogator 비교
│   ├── kosmos2_captioning.py        # Kosmos-2 모델 (3가지 버전)
│   ├── gpt4o_with_yolo.py           # GPT-4o + YOLO 결합 캡셔닝
│   ├── instructblip_with_yolo.py    # InstructBLIP + YOLO 결합
│   └── caption.ipynb                # 실험 노트북
│
├── 📂 computer_vision/              # 컴퓨터 비전 (객체 탐지)
│   ├── data_preprocessing.py        # 데이터 전처리 (캐시 삭제, 구조 정리)
│   ├── train_models.py              # YOLOv8 & Detectron2 학습
│   ├── test_and_evaluate.py         # 모델 테스트 및 평가 (mAP, CSV 저장)
│   ├── best.pt                      # 학습된 YOLOv8 모델
│   └── computer_vision.ipynb        # 실험 노트북
│
├── 📂 finetunning/                  # LLM 파인튜닝
│   ├── 📂 captioning/                # 이미지 캡션 데이터셋
│   │   ├── image_captions_blip.json
│   │   ├── image_captions_llava.json
│   │   └── image_captions_qwen.json
│   │
│   ├── 📂 layer_freezing/            # Layer Freezing 기법
│   │   ├── final_htp_model.ipynb
│   │   ├── HTP_data.jsonl
│   │   ├── interactive_test.ipynb
│   │   └── qwen2.5-htp-layer-freeze-final/
│   │
│   ├── 📂 LoRa/                      # LoRA 파인튜닝
│   │   ├── LoRa.ipynb
│   │   ├── HTP_data.jsonl
│   │   ├── htp_lora_model/
│   │   └── htp_merged_full_model/
│   │
│   ├── 📂 combined/                  # RAG + LLM 통합 시스템
│   │   ├── rag_model_combined.ipynb
│   │   ├── htp_rag_server.py        # FastAPI 서버
│   │   ├── simple_test_server.py
│   │   └── chroma_store/            # Vector DB
│   │
│   ├── Data_generation.ipynb         # 학습 데이터 생성
│   ├── test_base_model.ipynb         # 베이스 모델 테스트
│   └── model_comparison_results.csv  # 모델 성능 비교
│
├── 📂 RAG/                          # Retrieval-Augmented Generation
│   ├── 📂 Chunking/                  # 문서 청킹 전략
│   │   └── 그림_심리_멀티모달_RAG.ipynb
│   │
│   ├── 📂 Embedding/                 # 임베딩 파인튜닝
│   │   └── 심리_해석_임베딩_파인튜닝.ipynb
│   │
│   ├── 📂 Cross_Encoder/             # 재순위(Re-ranking) 모델
│   │   ├── BCE_cross_encoder.ipynb
│   │   ├── margin_cross_encoder.ipynb
│   │   └── 크로스_인코더_비교.ipynb
│   │
│   ├── 📂 LLM/                       # LLM 통합
│   │   ├── chatbot_model.ipynb
│   │   └── 멀티턴_멀티쿼리_history_RAG.ipynb
│   │
│   └── 📂 Web/                       # RAG 웹 API
│       ├── main.py                   # FastAPI 메인
│       ├── rag_engine.py             # RAG 엔진
│       └── embeddings.py             # 임베딩 처리
│
└── 📂 web/                          # 웹 애플리케이션
    ├── 📂 web_back-main/             # Backend (FastAPI)
    │   ├── multi_main.py             # 메인 서버
    │   ├── model.py                  # LLM 모델 로딩
    │   ├── rag_engine.py             # RAG 엔진
    │   ├── caption.py                # 이미지 캡셔닝
    │   └── Dockerfile
    │
    └── 📂 web_front-main/            # Frontend (React/Next.js)
        └── (웹 프론트엔드 파일들)
```
---
## [주요 기능]

**이미지 캡셔닝 (Image Captioning)**

HTP 그림 검사 이미지를 자연어로 설명하는 다양한 모델 비교

**지원 모델:**
- **BLIP** (Salesforce/blip-image-captioning-large)
- **InstructBLIP** (Salesforce/instructblip-flan-t5-xl)
- **Kosmos-2** (microsoft/kosmos-2-patch14-224)
- **CLIP Interrogator** (ViT-L-14/openai)
- **GPT-4o** (OpenAI Vision API)

**컴퓨터 비전 (Computer Vision)**

HTP 그림의 주요 요소(집, 나무, 사람)와 세부 객체 탐지

**모델:**
- **YOLOv8** (속도와 정확도 균형)
- **Detectron2** (Faster R-CNN R-50 FPN)
**학습 워크플로우:**
1. 데이터 전처리 (캐시 삭제, 라벨 구조 정리)
2. YOLOv8 학습 (100 epochs, early stopping)
3. Detectron2 학습 (2500 iterations, early stopping)

**이미지 캡션 모델+컴퓨터 비전 모델**
- YOLOv8로 객체 탐지 후 결과를 캡셔닝 모델의 프롬프트에 포함
- GPT-4o + YOLO, InstructBLIP + YOLO 조합으로 정확도 향상 시도 -> 캡션 모델 단독 출력 채택

**LLM 파인튜닝 (Fine-tuning)**

심리 해석 전문 LLM 구축을 위한 다양한 기법 실험

**파인튜닝 기법 비교**
- **Layer Freezing** | 하위 레이어 고정, 상위 레이어만 학습 | 빠른 학습, 메모리 효율적 | 제한적인 적응력 |
- **LoRA** | Low-Rank Adaptation | 적은 파라미터로 고품질 | 추가 어댑터 관리 필요 |
- **Full Fine-tuning** | 전체 파라미터 학습 | 최고 성능 | 높은 컴퓨팅 비용 |

**사용 모델**
- **Qwen 2.5 7B** (Alibaba - 다국어 지원 우수)
- **LLaVA** (멀티모달 - 이미지+텍스트)
- **BLIP-2** (이미지 캡셔닝 특화)

**RAG 시스템 (Retrieval-Augmented Generation)**

전문 심리학 지식 베이스를 활용한 정확한 해석

**파이프라인**
[사용자 질문] 
    ↓
[1. Chunking] 심리 서적/논문을 의미 단위로 분할
    ↓
[2. Embedding] BGE-M3 모델로 벡터화 (파인튜닝)
    ↓
[3. Retrieval] ChromaDB에서 유사 문서 검색
    ↓
[4. Re-ranking] Cross-Encoder로 최적화(**BCE Loss** vs **Margin Loss** 비교)
    ↓
[5. Generation] LLM이 컨텍스트 기반 답변 생성





---
## 🛠️ 기술 스택

### AI/ML
- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Computer Vision**: YOLOv8 (Ultralytics), Detectron2
- **LLM**: Qwen 2.5, LLaVA, GPT-4o
- **Embedding**: BGE-M3, Sentence-Transformers
- **Vector DB**: ChromaDB

### Backend
- **Framework**: FastAPI
- **API**: RESTful API
- **Authentication**: JWT

### Frontend
- **Framework**: React/Next.js
- **State Management**: Redux/Context API
- **Styling**: Tailwind CSS

### DevOps
- **Containerization**: Docker
---
