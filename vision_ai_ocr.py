import requests
import json
import base64
import argparse
import sys
from PIL import Image
import io

class VisionAIOCR:
    def __init__(self, ollama_url="http://localhost:11434"):
        print("AI 비전 OCR 시스템을 초기화하는 중...")
        
        # Ollama 설정
        self.ollama_url = ollama_url
        
        # 사용할 비전 모델들 (정확도 우선 순서로 배치)
        self.models = [
            "qwen2.5vl:7b",              # 7B 모델 최우선
            "llama3.2-vision:11b",        # 가장 큰 모델
            "llava-llama3:latest",        # 검증된 모델
            "qwen2.5vl:3b",              # 3B 모델
            "granite3.2-vision:latest"    # IBM의 경량 모델
        ]
        
        # 사용 가능한 모델 확인
        self.available_models = self.check_available_models()
        print(f"사용 가능한 모델: {self.available_models}")
        
        if not self.available_models:
            raise Exception("사용 가능한 비전 모델이 없습니다.")
        
        print("AI 비전 OCR 시스템 초기화 완료!")
    
    def check_available_models(self):
        """Ollama에서 사용 가능한 모델 확인"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                available = response.json().get('models', [])
                model_names = [model['name'] for model in available]
                return [model for model in self.models if model in model_names]
            else:
                print("Ollama 서버 연결 실패")
                return []
        except Exception as e:
            print(f"모델 확인 오류: {e}")
            return []
    
    def image_to_base64(self, image_path):
        """이미지를 base64로 변환"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"이미지 변환 오류: {e}")
            return None
    
    def extract_text_with_vision_ai(self, image_path, model_name, prompt_type="general"):
        """비전 AI 모델로 직접 OCR 수행"""
        try:
            print(f"\n{model_name} 모델로 OCR 수행 중... (프롬프트: {prompt_type})")
            
            # 연결 테스트
            test_response = requests.get(f"{self.ollama_url}/api/version", timeout=10)
            if test_response.status_code != 200:
                print("Ollama 서버 연결 실패")
                return None
            
            # 이미지를 base64로 변환
            image_base64 = self.image_to_base64(image_path)
            if not image_base64:
                return None
            
            # 프롬프트 선택
            if prompt_type == "table":
                prompt = """Read this Korean document image COMPLETELY and extract ALL text in EXACT VISUAL ORDER from top to bottom, left to right.

CRITICAL READING ORDER REQUIREMENTS:
- Follow the EXACT visual layout order - do NOT rearrange content
- Read from TOP to BOTTOM, LEFT to RIGHT as it appears visually
- Maintain the ORIGINAL sequence of all elements as they appear in the image
- Do NOT move paragraphs to the end - keep them in their visual position
- Do NOT group similar content types together - follow visual order

EXTRACTION REQUIREMENTS:
- Extract EVERY word, number, and symbol visible
- Output as plain text only (no table formatting)
- Include ALL Korean text, English text, numbers, and symbols
- Include ALL options (①, ②, ③, ④, ⑤) and their descriptions
- Include ALL explanatory paragraphs exactly where they appear visually
- Include ALL legal references and guidelines in their visual position

STRICT VISUAL ORDER PROCESS:
1. Start from the very top of the image
2. Read line by line, section by section as they appear
3. When you encounter a table, read it completely before moving to next visual element
4. When you encounter paragraph text, include it immediately in that position
5. Continue down the image maintaining exact visual sequence
6. Do NOT reorganize or reorder any content

CRITICAL: Keep ALL explanatory paragraphs in their EXACT visual position - do NOT move them to the end.

COMPLETENESS GUARANTEE:
- Continue reading until you have extracted EVERY SINGLE piece of text
- Do NOT stop until you reach the very bottom of the document
- Include ALL footnotes, references, and small text at the bottom
- Extract the COMPLETE document including all explanatory paragraphs starting with "각급기관의 장은..."

Extract everything maintaining perfect visual order and COMPLETE content:"""
            
            elif prompt_type == "detailed":
                prompt = """Extract ALL text from this Korean document in EXACT VISUAL ORDER.

CRITICAL VISUAL ORDER REQUIREMENTS:
1. Follow EXACT visual layout - do NOT rearrange any content
2. Read strictly from TOP to BOTTOM, LEFT to RIGHT as it appears
3. Maintain ORIGINAL sequence of all elements
4. Do NOT move any paragraphs or sections to different positions
5. Keep all explanatory text in its EXACT visual position

EXTRACTION REQUIREMENTS:
- Read EVERY piece of text in the document
- Include text in tables, headers, body content, and footnotes
- Pay attention to small text and numbers
- Read Korean characters accurately
- Include all punctuation, numbers, and special characters
- Use appropriate line breaks to separate sections
- Be completely comprehensive

STRICT PROCESS:
- Start from the very top of the image
- Read each visual element as you encounter it
- Do NOT skip ahead or reorganize content
- Maintain perfect sequential order

Extract ALL text maintaining exact visual order:"""
            
            else:  # general
                prompt = """Extract all Korean and English text from this image in EXACT VISUAL ORDER.

VISUAL ORDER REQUIREMENTS:
1. Follow the EXACT visual layout from top to bottom, left to right
2. Do NOT rearrange or reorder any content
3. Maintain the ORIGINAL position and sequence of all text
4. Keep paragraphs and sections in their visual position

FORMATTING REQUIREMENTS:
1. Output as plain text only (no table symbols like |, ---)
2. Use appropriate line breaks and paragraph separations
3. Accurately recognize numbers, symbols, and special characters
4. Extract both Korean and English text accurately
5. Do NOT use markdown table format
6. Preserve the logical structure and hierarchy

CRITICAL: Read in exact visual order - do NOT move any content to different positions.

Output only the extracted text maintaining perfect visual sequence:"""

            # API 호출 데이터
            data = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.0,        # 완전히 결정적인 출력
                    "top_p": 0.9,              # 더 많은 토큰 허용
                    "num_ctx": 32768,          # 컨텍스트 크기 대폭 증가
                    "num_predict": 8000,       # 출력 길이 추가 증가
                    "repeat_penalty": 1.0,     # 반복 방지 완전 해제
                    "repeat_last_n": 64,       # 반복 검사 범위 축소
                    "top_k": 100,              # 토큰 선택 다양성 증가
                    "min_p": 0.01,             # 최소 확률 임계값 낮춤
                    "stop": [],                # 정지 토큰 완전 제거
                    "presence_penalty": 0.0,   # 존재 페널티 제거
                    "frequency_penalty": 0.0   # 빈도 페널티 제거
                }
            }
            
            # 모델별 타임아웃 설정 (더 충분히 길게)
            if "qwen2.5vl:7b" in model_name:
                timeout = 1200  # 7B 모델은 20분
                print(f"AI 처리 시작... (qwen2.5vl:7b 모델, 예상 소요시간: 5-20분)")
            elif "llama3.2-vision:11b" in model_name:
                timeout = 1500  # 11B 모델은 25분
                print(f"AI 처리 시작... (대형 모델, 예상 소요시간: 10-25분)")
            elif "llava-llama3" in model_name:
                timeout = 900   # 15분
                print(f"AI 처리 시작... (예상 소요시간: 5-15분)")
            else:
                timeout = 720   # 12분
                print(f"AI 처리 시작... (예상 소요시간: 2-12분)")
            
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=data, 
                                   timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result.get('response', '').strip()
                if extracted_text:
                    print(f"✓ {model_name} 성공 (길이: {len(extracted_text)})")
                    return extracted_text
                else:
                    print(f"✗ {model_name} 빈 응답")
                    return None
            else:
                print(f"✗ {model_name} API 오류: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"✗ {model_name} 타임아웃")
            return None
        except Exception as e:
            print(f"✗ {model_name} 오류: {e}")
            return None
    
    def is_valid_ocr_result(self, text):
        """OCR 결과가 유효한지 내용으로 판단"""
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        # 의미없는 반복 패턴 검사
        if len(set(text.replace(' ', '').replace('%', '').replace('.', ''))) < 5:
            return False  # 너무 적은 종류의 문자만 사용
        
        # 숫자와 %만 있는 경우 (이전 오류 패턴)
        import re
        if re.match(r'^[\d\s.%]+$', text):
            return False
        
        # 한국어나 영어 단어가 포함되어 있는지 확인
        has_korean = bool(re.search(r'[가-힣]', text))
        has_english_words = bool(re.search(r'[a-zA-Z]{2,}', text))
        
        if has_korean or has_english_words:
            return True
        
        # 최소 길이 확인 (의미있는 내용이라면)
        if len(text) < 10:
            return False
            
        return True
    
    def analyze_image_content(self, image_path):
        """이미지 내용을 분석하여 최적의 프롬프트 전략 결정"""
        try:
            print("이미지 내용 분석 중...")
            
            # 가장 빠른 모델로 이미지 분석 (작은 모델들 우선)
            analysis_models = ["granite3.2-vision:latest", "qwen2.5vl:3b"]
            fastest_model = None
            
            for model in analysis_models:
                if model in self.available_models:
                    fastest_model = model
                    break
            
            if not fastest_model:
                fastest_model = self.available_models[-1]  # 마지막 모델 사용
            
            print(f"분석용 모델: {fastest_model}")
            
            image_base64 = self.image_to_base64(image_path)
            if not image_base64:
                return "general"
            
            # 이미지 분석용 프롬프트
            analysis_prompt = """Analyze this image and describe its layout and content type briefly.

Focus on:
1. Is this a table/structured document with rows and columns?
2. Is this a form with fields and labels?  
3. Is this regular text/paragraph content?
4. Are there multiple sections or complex layouts?
5. Is the text densely packed or sparse?

Respond with just one of these categories: TABLE, FORM, PARAGRAPH, COMPLEX, or SIMPLE"""

            data = {
                "model": fastest_model,
                "prompt": analysis_prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 2048,   # 분석용이므로 작은 컨텍스트
                    "num_predict": 50  # 매우 짧은 응답만 필요
                }
            }
            
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=data, 
                                   timeout=30)  # 분석은 빠르게
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', '').strip().upper()
                
                # 분석 결과에 따른 전략 결정
                if "TABLE" in analysis:
                    strategy = "table"
                    reason = "표/테이블 구조 감지"
                elif "FORM" in analysis:
                    strategy = "detailed"
                    reason = "양식/폼 구조 감지"
                elif "COMPLEX" in analysis:
                    strategy = "detailed"
                    reason = "복잡한 레이아웃 감지"
                elif "PARAGRAPH" in analysis:
                    strategy = "general"
                    reason = "일반 문서 텍스트 감지"
                else:
                    strategy = "detailed"
                    reason = "기본 상세 분석 선택"
                
                print(f"✓ 이미지 분석 완료: {analysis}")
                print(f"✓ 선택된 전략: {strategy} ({reason})")
                return strategy
                
            else:
                print("이미지 분석 실패, 기본 전략 사용")
                return "detailed"
                
        except Exception as e:
            print(f"이미지 분석 오류: {e}")
            return "detailed"
    
    def process_image_with_smart_strategy(self, image_path):
        """이미지 분석 후 최적 전략으로 OCR 수행"""
        results = {}
        
        print(f"지능형 이미지 분석 시작: {image_path}")
        print("=" * 60)
        
        # 1단계: 이미지 내용 분석
        optimal_strategy = self.analyze_image_content(image_path)
        
        # 2단계: 최적 전략으로 OCR 수행 (qwen2.5vl:7b 최우선)
        ocr_models = ["qwen2.5vl:7b", "qwen2.5vl:3b", "llava-llama3:latest", "llama3.2-vision:11b"]  # 7B 모델 최우선
        primary_model = None
        
        for model in ocr_models:
            if model in self.available_models:
                primary_model = model
                break
        
        if not primary_model:
            primary_model = self.available_models[0]
        
        print(f"OCR용 모델: {primary_model}")
        print(f"타임아웃 없이 완료까지 대기합니다...")
        
        print(f"\n주 전략({optimal_strategy})으로 OCR 수행:")
        result = self.extract_text_with_vision_ai(image_path, primary_model, optimal_strategy)
        
        if self.is_valid_ocr_result(result):
            print(f"✓ OCR 완료! (길이: {len(result)})")
            return result, f"{primary_model} ({optimal_strategy})"
        
        # 결과가 부족하면 선택된 전략으로 모든 모델을 시도
        print(f"\n주 모델이 실패했습니다. {optimal_strategy} 전략으로 다른 모델들을 시도합니다:")
        
        all_results = {}
        
        for model in self.available_models:
            if model == primary_model:
                continue  # 이미 시도한 모델은 스킵
                
            print(f"\n=== {model} 모델 ({optimal_strategy} 전략) 시도 ===")
            result = self.extract_text_with_vision_ai(image_path, model, optimal_strategy)
            
            if self.is_valid_ocr_result(result):
                print(f"✓ {model} 모델에서 성공적으로 추출했습니다! (길이: {len(result)})")
                return result, f"{model}_{optimal_strategy}"
            else:
                print(f"실패 또는 유효하지 않은 결과 (길이: {len(result) if result else 0})")
        
        # 모든 모델이 실패하면 백업 전략 시도
        print(f"\n모든 모델이 {optimal_strategy} 전략으로 실패했습니다. 백업 전략을 시도합니다:")
        
        # 백업 전략 결정
        if optimal_strategy == "table":
            backup_strategy = "detailed"
        elif optimal_strategy == "detailed":
            backup_strategy = "table"
        else:
            backup_strategy = "detailed"
        
        print(f"\n백업 전략: {backup_strategy}")
        
        for model in self.available_models:
            print(f"\n=== {model} 모델 ({backup_strategy} 전략) 시도 ===")
            result = self.extract_text_with_vision_ai(image_path, model, backup_strategy)
            
            if self.is_valid_ocr_result(result):
                print(f"✓ {model} 모델에서 백업 전략으로 성공!")
                return result, f"{model}_{backup_strategy}"
            else:
                print(f"실패 또는 유효하지 않은 결과 (길이: {len(result) if result else 0})")
        
        print("모든 모델과 전략에서 OCR 실패")
        return None, None
    
    def process_image(self, image_path, output_file=None):
        """메인 이미지 처리 함수"""
        print(f"AI 비전 OCR 처리 시작")
        print(f"입력 이미지: {image_path}")
        
        # 지능형 전략으로 텍스트 추출
        extracted_text, used_strategy = self.process_image_with_smart_strategy(image_path)
        
        if not extracted_text:
            print("텍스트 추출 실패")
            return None
        
        # 결과 출력
        print("\n" + "=" * 60)
        print(f"최종 추출 결과 (전략: {used_strategy})")
        print("=" * 60)
        print(extracted_text)
        print("=" * 60)
        
        # 파일 저장
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== 지능형 AI 비전 OCR 결과 ===\n")
                f.write(f"사용 전략: {used_strategy}\n")
                f.write(f"추출 시간: {__import__('datetime').datetime.now()}\n")
                f.write("\n" + "=" * 50 + "\n\n")
                f.write(extracted_text)
            print(f"\n결과가 {output_file}에 저장되었습니다.")
        
        return extracted_text

def main():
    parser = argparse.ArgumentParser(description='AI Vision-based OCR System')
    parser.add_argument('image_path', help='입력 이미지 파일 경로')
    parser.add_argument('-o', '--output', help='출력 텍스트 파일 경로 (선택사항)')
    parser.add_argument('--ollama-url', default='http://localhost:11434', 
                       help='Ollama 서버 URL (기본값: http://localhost:11434)')
    
    args = parser.parse_args()
    
    try:
        # OCR 시스템 생성
        ocr = VisionAIOCR(ollama_url=args.ollama_url)
        
        # 이미지 처리
        result = ocr.process_image(args.image_path, args.output)
        
        if result is None:
            sys.exit(1)
            
    except Exception as e:
        print(f"시스템 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
