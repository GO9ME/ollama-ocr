# Korean OCR System

AI 기반 한국어 문서 OCR 시스템입니다. 다양한 AI 비전 모델을 활용하여 한국어 텍스트를 정확하게 추출합니다.

## 주요 기능

- **지능형 이미지 분석**: 문서 유형을 자동 감지하여 최적의 추출 전략 선택
- **다중 모델 지원**: qwen2.5vl, llama3.2-vision, llava-llama3, granite3.2-vision 등
- **순서 보장**: 문서의 시각적 레이아웃 순서를 정확히 유지
- **완전성 보장**: 문서의 모든 텍스트를 누락 없이 추출
- **표 구조 지원**: 복잡한 표 형태 문서의 정확한 텍스트 추출

## 파일 구성

### 주요 시스템
- **`vision_ai_ocr.py`** - 메인 AI 비전 OCR 시스템 (권장)

### 설정 파일
- **`requirements.txt`** - 필요한 Python 패키지 목록

## 설치 및 설정

### 1. 환경 준비
```bash
# 가상환경 생성 및 활성화
python -m venv ocr_env
source ocr_env/bin/activate  # Linux/Mac
# 또는
ocr_env\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. Ollama 설정 (vision_ai_ocr.py 사용시)
```bash
# Ollama 설치 후 필요한 모델 다운로드
ollama pull qwen2.5vl:7b
ollama pull llama3.2-vision:11b
ollama pull llava-llama3:latest
ollama pull granite3.2-vision:latest
```

## 사용법

### AI 비전 OCR (권장)
```bash
python vision_ai_ocr.py image.jpg -o output.txt
```

**옵션:**
- `-o, --output`: 출력 텍스트 파일 경로
- `--ollama-url`: Ollama 서버 URL (기본값: http://192.168.50.123:11434)


## 시스템 특징

### vision_ai_ocr.py (권장 시스템)

**장점:**
- 가장 정확한 한국어 텍스트 인식
- 지능형 문서 유형 감지
- 표 구조 문서 특화 처리
- 완전한 텍스트 추출 보장
- 시각적 순서 정확히 유지

**처리 과정:**
1. 이미지 내용 자동 분석
2. 문서 유형에 따른 최적 전략 선택
3. 다중 모델 백업 시스템
4. 품질 검증 및 결과 출력

**지원 전략:**
- **table**: 표/테이블 구조 문서
- **detailed**: 복잡한 레이아웃 문서
- **general**: 일반 텍스트 문서

### 모델별 특성

| 모델 | 크기 | 정확도 | 속도 | 용도 |
|------|------|--------|------|------|
| qwen2.5vl:7b | 7B | 높음 | 중간 | 메인 OCR (최우선) |
| llama3.2-vision:11b | 11B | 최고 | 느림 | 고품질 추출 |
| llava-llama3:latest | 7B | 높음 | 빠름 | 범용 OCR |
| granite3.2-vision:latest | 3B | 중간 | 빠름 | 이미지 분석 |

## 설정 파라미터

### 생성 파라미터 (vision_ai_ocr.py)
```python
"options": {
    "temperature": 0.0,        # 결정적 출력
    "num_ctx": 32768,          # 큰 컨텍스트
    "num_predict": 8000,       # 긴 출력 허용
    "repeat_penalty": 1.0,     # 반복 방지 해제
    "stop": []                 # 정지 토큰 제거
}
```

### 타임아웃 설정
- qwen2.5vl:7b: 20분
- llama3.2-vision:11b: 25분
- llava-llama3: 15분
- 기타: 12분

## 문제 해결

### 일반적인 문제

**1. 텍스트가 잘리는 경우**
- `num_predict` 값을 증가 (현재: 8000)
- 타임아웃 시간 확인

**2. 순서가 뒤바뀌는 경우**
- vision_ai_ocr.py 사용 (순서 보장 기능)
- "EXACT VISUAL ORDER" 프롬프트 적용됨

**3. 한국어 인식 오류**
- qwen2.5vl:7b 모델 우선 사용
- EasyOCR 대신 AI 비전 모델 사용

**4. 모델 연결 실패**
```bash
# Ollama 서버 상태 확인
ollama list
ollama serve  # 서버 시작
```

### 성능 최적화

**메모리 부족시:**
- 더 작은 모델 사용 (qwen2.5vl:3b)
- 이미지 크기 조정

**속도 향상:**
- granite3.2-vision 모델 사용
- 분석 단계 생략 (`detailed` 전략 직접 사용)

## 개발 히스토리

1. **ko-trocr 기반** → 한글 자모 분해 문제
2. **EasyOCR** → 표 구조 인식 한계  
3. **EasyOCR + AI 후처리** → 정확도 개선
4. **순수 AI 비전** → 최고 성능 달성
5. **지능형 전략 선택** → 문서 유형별 최적화
6. **순서 보장 시스템** → 시각적 레이아웃 유지

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 기여

버그 리포트나 기능 제안은 이슈로 등록해 주세요.