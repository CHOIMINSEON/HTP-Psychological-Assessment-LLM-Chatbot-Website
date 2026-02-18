"""
YOLO 및 Detectron2 학습을 위한 데이터 전처리 및 구조 정리
- 캐시 파일 삭제
- 라벨 폴더 구조 정리
- YOLO 하위 폴더 파일 이동
필요한 패키지: os, shutil
"""

import os
import shutil


def delete_cache_files(data_root):
    """데이터셋 내 모든 .cache 파일 삭제"""
    print("🧹 캐시 파일 삭제 중...")
    
    splits = ["train", "val", "test"]
    deleted_count = 0
    
    for split in splits:
        labels_dir = os.path.join(data_root, split, "labels")
        if not os.path.exists(labels_dir):
            continue
        
        for root, dirs, files in os.walk(labels_dir):
            for file in files:
                if file.endswith(".cache"):
                    cache_path = os.path.join(root, file)
                    try:
                        os.remove(cache_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"⚠️ 캐시 삭제 실패: {cache_path} - {e}")
    
    print(f"✅ 총 {deleted_count}개의 캐시 파일 삭제 완료")


def reorganize_label_structure(data_root):
    """라벨 폴더 구조 정리: YOLO 하위 폴더의 파일을 상위로 이동"""
    print("\n📁 라벨 구조 정리 중...")
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        labels_dir = os.path.join(data_root, split, "labels")
        if not os.path.exists(labels_dir):
            print(f"⚠️ 경로가 존재하지 않음: {labels_dir}")
            continue
        
        print(f"\n🔍 정리 중: {labels_dir}")
        
        # labels/ 하위 폴더 탐색 (예: 나무, 집, 사람 등)
        for class_dir in os.listdir(labels_dir):
            class_path = os.path.join(labels_dir, class_dir)
            
            if not os.path.isdir(class_path):
                continue
            
            yolo_subdir = os.path.join(class_path, "YOLO")
            
            # YOLO 하위 폴더가 존재할 경우 처리
            if os.path.isdir(yolo_subdir):
                txt_files = [f for f in os.listdir(yolo_subdir) if f.endswith(".txt")]
                
                if txt_files:
                    print(f"  📁 {class_dir}/YOLO → {len(txt_files)}개 파일 이동 중...")
                    
                    # YOLO 폴더 안의 모든 txt를 한 단계 위로 이동
                    for f in txt_files:
                        src = os.path.join(yolo_subdir, f)
                        dst = os.path.join(class_path, f)
                        try:
                            shutil.move(src, dst)
                        except Exception as e:
                            print(f"    ⚠️ 파일 이동 실패: {f} - {e}")
                    
                    # YOLO 폴더 삭제
                    try:
                        shutil.rmtree(yolo_subdir)
                    except Exception as e:
                        print(f"    ⚠️ YOLO 폴더 삭제 실패: {yolo_subdir} - {e}")
        
        # .cache 파일 삭제 (다시 한번 확인)
        for cache_file in os.listdir(labels_dir):
            if cache_file.endswith(".cache"):
                cache_path = os.path.join(labels_dir, cache_file)
                try:
                    os.remove(cache_path)
                    print(f"  🗑️ 캐시 삭제: {cache_file}")
                except Exception as e:
                    print(f"  ⚠️ 캐시 삭제 실패: {cache_file} - {e}")
    
    print("\n✅ 모든 라벨 구조 정리 완료!")


def verify_data_structure(data_root):
    """데이터 구조 검증"""
    print("\n🔍 데이터 구조 검증 중...")
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        images_dir = os.path.join(data_root, split, "images")
        labels_dir = os.path.join(data_root, split, "labels")
        
        if os.path.exists(images_dir):
            image_count = sum([len(files) for _, _, files in os.walk(images_dir)])
            print(f"✅ {split}/images: {image_count}개 파일")
        else:
            print(f"⚠️ {split}/images: 경로 없음")
        
        if os.path.exists(labels_dir):
            label_count = sum([len([f for f in files if f.endswith('.txt')]) 
                             for _, _, files in os.walk(labels_dir)])
            print(f"✅ {split}/labels: {label_count}개 txt 파일")
        else:
            print(f"⚠️ {split}/labels: 경로 없음")


def main(data_root):
    """메인 실행 함수"""
    print("=" * 50)
    print("데이터 전처리 시작")
    print("=" * 50)
    print(f"데이터 루트: {data_root}\n")
    
    # 1. 캐시 파일 삭제
    delete_cache_files(data_root)
    
    # 2. 라벨 구조 정리
    reorganize_label_structure(data_root)
    
    # 3. 데이터 구조 검증
    verify_data_structure(data_root)
    
    print("\n" + "=" * 50)
    print("데이터 전처리 완료")
    print("=" * 50)


if __name__ == "__main__":
    # 데이터 루트 경로 설정 (사용자가 수정해야 함)
    DATA_ROOT = "/content/drive/MyDrive/Colab/T_V_T"
    
    # 경로 존재 확인
    if not os.path.exists(DATA_ROOT):
        print(f"❌ 오류: 데이터 루트 경로를 찾을 수 없음: {DATA_ROOT}")
        print("경로를 확인하고 DATA_ROOT 변수를 수정해주세요.")
    else:
        main(DATA_ROOT)
