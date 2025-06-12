import os
import base64
import io
import gc
import re
from PIL import Image
from typing import List, Dict, Optional, Tuple, Union

def resize_image(image_path, max_size=(600, 600)):
    """이미지 크기 조정"""
    # 파일 존재 여부 확인
    if not os.path.exists(image_path):
        print(f"이미지 파일이 존재하지 않음: {image_path}")
        return None
        
    try:
        with Image.open(image_path) as img:
            # 이미지가 이미 충분히 작은 경우 그대로 반환
            if img.width <= max_size[0] and img.height <= max_size[1]:
                # 복사본 만들기 (원본 객체가 아닌 복사본 반환)
                return img.copy()
            
            # 비율 유지하면서 크기 조정
            img_copy = img.copy()
            img_copy.thumbnail(max_size)
            
            # 메모리 사용량 최적화를 위해 RGB 모드로 변환 (알파 채널 제거)
            if img_copy.mode == 'RGBA':
                background = Image.new('RGB', img_copy.size, (255, 255, 255))
                background.paste(img_copy, mask=img_copy.split()[3])  # 알파 채널을 마스크로 사용
                return background
            elif img_copy.mode != 'RGB':
                return img_copy.convert('RGB')
            
            return img_copy
    except Exception as e:
        print(f"이미지 크기 조정 중 오류: {str(e)}")
        return None

def encode_image_to_base64(image_path, resize=True, max_size=(600, 600)):
    """이미지를 Base64로 인코딩, 선택적으로 크기 조정"""
    # 파일 존재 여부 먼저 확인
    if not os.path.exists(image_path):
        print(f"이미지 파일이 존재하지 않음: {image_path}")
        return None
        
    try:
        if resize:
            img = resize_image(image_path, max_size=max_size)
            if img:
                # 메모리에 이미지 저장 (JPEG 형식으로 변환하여 크기 감소)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85, optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # 메모리 정리
                buffer.close()
                del img
                gc.collect()
                
                return encoded
        
        # 크기 조정 실패하거나 resize=False인 경우 원본 사용
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"이미지 인코딩 중 오류: {str(e)}")
        return None

def extract_image_paths(text: str) -> List[str]:
    """Ollama 응답에서 이미지 경로 추출
    
    Args:
        text: Ollama 응답 텍스트
        
    Returns:
        추출된 이미지 경로 목록
    """
    image_paths = []
    
    # 디버깅을 위해 전체 텍스트 출력
    print("응답 텍스트 분석 중...")
    
    # 가장 단순한 방법: 전체 텍스트에서 절대 경로 패턴 검색
    pattern = re.compile(r'(/[\w/._-]+\.(?:png|jpg|jpeg|gif|bmp))', re.IGNORECASE)
    
    # 전체 텍스트에서 경로 검색
    for match in pattern.finditer(text):
        path = match.group(1)
        # 경로가 유효한지 확인
        if os.path.exists(path):
            print(f"이미지 파일 발견: {path}")
            image_paths.append(path)
        else:
            print(f"이미지 파일이 존재하지 않음: {path}")
    
    # 결과 확인
    if not image_paths:
        print("추출된 이미지 경로가 없습니다.")
    else:
        print(f"추출된 이미지 경로: {len(image_paths)}개")
    
    return image_paths

def process_response_with_images(response: str, remove_image_section: bool = True) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """Ollama 응답을 처리하고 관련 이미지를 base64로 인코딩
    
    Args:
        response: Ollama 응답 텍스트
        remove_image_section: 'Relevant Images:' 섹션을 응답에서 제거할지 여부
        
    Returns:
        처리된 응답과 base64로 인코딩된 이미지 정보를 포함하는 딕셔너리
    """
    # 원본 응답 보관
    original_response = response
    
    # 이미지 경로 추출
    image_paths = extract_image_paths(response)
    
    # 이미지 관련 섹션 처리
    cleaned_response = response
    if remove_image_section and image_paths:
        # 1. 'Relevant Images:' 섹션 찾기 및 처리
        pattern1 = re.compile(r'\n\s*Relevant\s+Images?(?:\s*:|\s*파일\s+경로:)?\s*\n([^\n]*(?:\n(?!\n)[^\n]*)*)', re.IGNORECASE | re.DOTALL)
        match1 = pattern1.search(response)
        
        if match1:
            # 섹션 제거 후 '관련 이미지'로 대체
            section_with_header = match1.group(0)  # 헤더와 내용 포함
            cleaned_response = response.replace(section_with_header, '\n\n관련 이미지\n')
            print("'Relevant Images:' 섹션을 '관련 이미지'로 대체했습니다.")
        
        # 2. '관련 이미지 파일 경로:' 섹션 찾기 및 처리
        pattern2 = re.compile(r'\n\s*관련\s+이미지\s+파일\s+경로:\s*\n([^\n]*(?:\n(?!\n)[^\n]*)*)', re.DOTALL)
        match2 = pattern2.search(cleaned_response)
        
        if match2:
            # 섹션 제거 후 '관련 이미지'로 대체
            section_with_header = match2.group(0)  # 헤더와 내용 포함
            cleaned_response = cleaned_response.replace(section_with_header, '\n\n관련 이미지\n')
            print("'관련 이미지 파일 경로:' 섹션을 '관련 이미지'로 대체했습니다.")
        
        # 3. 일반 '관련 이미지' 섹션이 있는지 확인
        pattern3 = re.compile(r'\n\s*관련\s+이미지\s*(?::|\n)([^\n]*(?:\n(?!\n)[^\n]*)*)', re.DOTALL)
        match3 = pattern3.search(cleaned_response)
        
        # 이미 '관련 이미지' 섹션이 있는 경우, 이미지 경로 제거
        if match3:
            section_with_content = match3.group(0)  # 헤더와 내용 포함
            header_only = '\n\n관련 이미지\n'
            cleaned_response = cleaned_response.replace(section_with_content, header_only)
            print("'관련 이미지' 섹션에서 이미지 경로를 제거했습니다.")
        # '관련 이미지' 섹션이 없는 경우, 새로 추가
        elif not match1 and not match2:
            # 응답 끝에 '관련 이미지' 추가
            cleaned_response = cleaned_response.rstrip() + '\n\n관련 이미지\n'
            print("'관련 이미지' 섹션을 새로 추가했습니다.")
        
        # 4. 개별 파일 경로 텍스트 제거
        for path in image_paths:
            # 파일 경로: /path/to/image.png 형태 제거
            path_pattern1 = re.compile(r'\n\s*파일\s+경로\s*:\s*' + re.escape(path) + r'\s*', re.DOTALL)
            cleaned_response = re.sub(path_pattern1, '\n', cleaned_response)
            
            # 경로만 있는 경우 제거
            path_pattern2 = re.compile(r'\n\s*' + re.escape(path) + r'\s*', re.DOTALL)
            cleaned_response = re.sub(path_pattern2, '\n', cleaned_response)
            
            print(f"파일 경로 텍스트 제거: {path}")
        
        # 연속된 줄바꿈 제거 및 정리
        cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response)
        cleaned_response = re.sub(r'\n\s*관련\s+이미지\s*\n+\s*관련\s+이미지\s*\n', '\n\n관련 이미지\n', cleaned_response)
    
    result = {
        "response": cleaned_response,
        "original_response": original_response,
        "images": []
    }
    
    if not image_paths:
        print("추출된 이미지 경로가 없습니다.")
        return result
    
    print(f"추출된 이미지 경로: {len(image_paths)}개")
    
    # 각 이미지를 base64로 인코딩
    for image_path in image_paths:
        try:
            print(f"이미지 인코딩 시도: {image_path}")
            
            # 파일 존재 여부 한 번 더 확인
            if not os.path.exists(image_path):
                print(f"이미지 파일이 존재하지 않음: {image_path}")
                continue
                
            # 이미지 확장자 확인
            if not any(image_path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]):
                print(f"지원되지 않는 이미지 파일 형식: {image_path}")
                continue
            
            base64_image = encode_image_to_base64(image_path)
            if base64_image:
                # 파일명 추출
                filename = os.path.basename(image_path)
                
                # 파일명에서 title, index, page_num 추출 (119_1_Picture_1.00.png 형식)
                parts = filename.split('_')
                title = parts[0] if len(parts) > 0 else ''
                index = parts[1] if len(parts) > 1 else ''
                
                # page_num 추출 (Picture_1.00.png 형식)
                page_info = '_'.join(parts[2:]) if len(parts) > 2 else ''
                page_parts = page_info.split('.')
                page_num = page_parts[0] if len(page_parts) > 0 else ''
                
                image_info = {
                    "path": image_path,
                    "filename": filename,
                    "base64": base64_image,
                    "title": title,
                    "index": index,
                    "page_num": page_num
                }
                result["images"].append(image_info)
                print(f"이미지 인코딩 성공: {image_path}")
            else:
                print(f"이미지 인코딩 실패: {image_path}")
        except Exception as e:
            print(f"이미지 처리 중 예외 발생: {image_path}, 오류: {str(e)}")
    
    return result
