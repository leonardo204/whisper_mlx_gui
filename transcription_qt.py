"""
전사 관리 클래스 - Qt 이벤트 시스템 사용 버전
- 기존 transcription.py의 기능을 이벤트 드리븐 방식으로 리팩토링
"""

import numpy as np
import os
import sys
import time
import tempfile
import json
import re
from typing import Dict, Optional, List, Union, Tuple
import threading
from collections import deque
import datetime
from PySide6.QtCore import QObject, Signal, Slot, QThread

from deep_translator import GoogleTranslator
from logging_utils import LogManager
from events import event_bus

# 모델 지원 언어
SUPPORTED_LANGUAGES = {
    'ko': '한국어',
    'en': '영어',
    'ja': '일본어',
    'zh': '중국어',
    'es': '스페인어',
    'fr': '프랑스어',
    'de': '독일어',
    'it': '이탈리아어',
    'ru': '러시아어',
    'ar': '아랍어',
    'hi': '힌디어',
    'pt': '포르투갈어',
    'nl': '네덜란드어',
    'tr': '터키어',
    'pl': '폴란드어',
    'vi': '베트남어',
    'th': '태국어',
    'id': '인도네시아어'
}


class TextProcessor(QObject):
    """
    텍스트 처리 클래스 (Qt 버전)
    - 텍스트 정제
    - 불완전한 문장 처리
    - 반복 제거
    - 문장 병합
    """
    def __init__(self):
        """텍스트 프로세서 초기화"""
        super().__init__()
        self.logger = LogManager()
        self.logger.log_info("텍스트 프로세서 초기화")
        
        # 최근 처리 텍스트 기록 (중복 검사용)
        self.recent_texts = deque(maxlen=5)
        
        # 정제 패턴
        self.cleanup_patterns = [
            (r'\s+', ' '),                # 연속된 공백 정규화
            (r'[\s.,!?]+$', ''),          # 끝부분 특수문자 제거
            (r'^[\s.,!?]+', ''),          # 시작부분 특수문자 제거
            (r'[.]{2,}', '...'),          # 마침표 정규화
            (r'[,]{2,}', ','),            # 쉼표 정규화
            (r'[!]{2,}', '!!'),           # 느낌표 정규화
            (r'[?]{2,}', '??')            # 물음표 정규화
        ]
        
        # 한국어 문장 종결 패턴
        self.ko_sentence_end = r'[.!?~…]\s*$|[다요죠양함임니까까요까봐봐죠네요네죠]\s*$'
        
        # 단어 반복 패턴 (단일 단어 반복)
        self.word_repetition_pattern = r'(\b\w+\b)(\s+\1\b)+'
        
        # 구문 반복 패턴 (2-5단어 구문 반복)
        self.phrase_repetition_patterns = [
            # 2단어 구문 반복 패턴
            r'(\b\w+\s+\w+\b)(\s+\1\b)+',
            # 3단어 구문 반복 패턴
            r'(\b\w+\s+\w+\s+\w+\b)(\s+\1\b)+',
            # 4단어 구문 반복 패턴
            r'(\b\w+\s+\w+\s+\w+\s+\w+\b)(\s+\1\b)+',
            # 5단어 구문 반복 패턴
            r'(\b\w+\s+\w+\s+\w+\s+\w+\s+\w+\b)(\s+\1\b)+'
        ]
        
        # 한국어 반복 패턴 (2-5어절 구문 반복)
        self.korean_repetition_patterns = [
            # 2어절 구문 반복
            r'([가-힣]+\s+[가-힣]+)(\s+\1)+',
            # 3어절 구문 반복
            r'([가-힣]+\s+[가-힣]+\s+[가-힣]+)(\s+\1)+',
            # 4어절 구문 반복
            r'([가-힣]+\s+[가-힣]+\s+[가-힣]+\s+[가-힣]+)(\s+\1)+',
            # 5어절 구문 반복
            r'([가-힣]+\s+[가-힣]+\s+[가-힣]+\s+[가-힣]+\s+[가-힣]+)(\s+\1)+'
        ]
        
        self.logger.log_info("텍스트 프로세서가 초기화되었습니다")
        event_bus.status.log.emit("info", "텍스트 프로세서가 초기화되었습니다")

    def process_text(self, text: str) -> Optional[str]:
        """전사 텍스트 처리"""
        try:
            if not text or not text.strip():
                return None
                
            # 기본 정제
            processed = text.strip()
            
            # 정규화 패턴 적용
            for pattern, replacement in self.cleanup_patterns:
                processed = re.sub(pattern, replacement, processed)
                
            # 단일 단어 반복 제거
            processed = re.sub(self.word_repetition_pattern, r'\1', processed)
            
            # 영어 구문 반복 제거
            for pattern in self.phrase_repetition_patterns:
                processed = re.sub(pattern, r'\1', processed)
                
            # 한국어 구문 반복 제거
            for pattern in self.korean_repetition_patterns:
                processed = re.sub(pattern, r'\1', processed)
                
            # 한국어 반복 구문 추가 검사 (정규식으로 감지하기 어려운 경우)
            processed = self._remove_korean_repetitions(processed)
            
            # 이전 텍스트와 유사도 확인 (중복 방지)
            if self._is_duplicate(processed):
                self.logger.log_debug(f"중복 텍스트 감지됨: {processed}")
                return None
                
            # 결과 저장 및 반환
            self.recent_texts.append(processed)
            return processed
            
        except Exception as e:
            self.logger.log_error("text_processing", f"텍스트 처리 중 오류: {str(e)}")
            event_bus.status.error.emit("text_processing_error", "텍스트 처리 중 오류", str(e))
            return text  # 오류 시 원본 반환

    def _remove_korean_repetitions(self, text: str) -> str:
        """한국어 반복 구문 추가 검사 및 제거"""
        try:
            # 1. 단순하고 짧은 단어가 많이 반복되는 경우 (예: "정치권의" 반복)
            # 간단한 반복 정규식 추가 - 3회 이상 반복되는 동일 단어 패턴
            simple_repeat_pattern = r'(\S+)(\s+\1){2,}'
            text = re.sub(simple_repeat_pattern, r'\1', text)

            words = text.split()
            if len(words) < 6:  # 적은 단어 수는 처리 불필요
                return text
                
            # 반복 윈도우 크기 (2-6어절)
            for window_size in range(2, 7):
                if len(words) < window_size * 2:  # 윈도우 크기의 2배 이상 단어가 있어야 함
                    continue
                    
                i = 0
                result = []
                skip_to = -1
                
                while i < len(words):
                    if i < skip_to:
                        i += 1
                        continue
                        
                    # 현재 위치에서 윈도우 크기만큼의 단어들
                    curr_window = words[i:i+window_size]
                    
                    # 반복 감지
                    repetition_found = False
                    for j in range(i + window_size, len(words) - window_size + 1, window_size):
                        next_window = words[j:j+window_size]
                        
                        # 윈도우가 동일한지 확인
                        if curr_window == next_window:
                            if not repetition_found:  # 첫 번째 윈도우는 저장
                                result.extend(curr_window)
                                repetition_found = True
                            
                            skip_to = j + window_size  # 반복된 부분 건너뛰기
                        else:
                            break
                            
                    if not repetition_found:
                        result.append(words[i])
                        i += 1
                    else:
                        i = skip_to
                
                words = result  # 다음 윈도우 크기 처리를 위해 결과 업데이트
                
            return ' '.join(words)
            
        except Exception as e:
            self.logger.log_error("korean_repetition", f"한국어 반복 구문 처리 중 오류: {str(e)}")
            return text  # 오류 시 원본 반환

    def _is_duplicate(self, text: str) -> bool:
        """텍스트 중복 여부 확인"""
        if not self.recent_texts:
            return False

        # 정확히 일치하는 경우
        if text in self.recent_texts:
            return True

        # 일부만 다른 경우 (80% 이상 유사)
        latest = self.recent_texts[-1]

        # 길이가 크게 다르면 중복 아님
        if abs(len(text) - len(latest)) > min(len(text), len(latest)) * 0.5:
            return False

        # 간단한 유사도 계산 (자카드 유사도)
        words1 = set(text.split())
        words2 = set(latest.split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        similarity = intersection / union if union > 0 else 0
        return similarity > 0.8  # 80% 이상 유사하면 중복으로 간주

    def is_complete_sentence(self, text: str) -> bool:
        """문장 완성도 확인"""
        # 한국어 문장 종결 확인
        if re.search(self.ko_sentence_end, text):
            return True

        # 영어 문장 종결 확인
        if re.search(r'[.!?]\s*$', text):
            return True

        # 최소 길이 기준
        if len(text) > 30:  # 긴 문장은 완성으로 간주
            return True

        return False

    def combine_texts(self, texts: List[str]) -> str:
        """여러 텍스트 조각을 하나로 결합"""
        if not texts:
            return ""

        # 중복 제거
        unique_texts = []
        for text in texts:
            if text and text not in unique_texts:
                unique_texts.append(text)

        # 결합
        return " ".join(unique_texts)


class TranscriptionThread(QThread):
    """
    전사 작업 스레드
    - 별도 스레드에서 Whisper 모델 실행
    - 결과를 시그널로 반환
    """
    # 전사 결과 시그널
    transcription_result = Signal(dict)
    
    # 오류 발생 시그널
    error = Signal(str, str)
    
    # 진행 상황 시그널
    progress = Signal(int, float)  # (세그먼트 ID, 진행률)
    
    def __init__(self, segment: Dict, model, use_faster_whisper: bool = True, use_mlx: bool = False):
        """전사 스레드 초기화"""
        super().__init__()
        self.logger = LogManager()
        self.segment = segment
        self.model = model
        self.use_faster_whisper = use_faster_whisper
        self.use_mlx = use_mlx
        self.segment_id = segment.get('id', 0)
    
    def run(self):
        """스레드 실행"""
        try:
            # 전사 시작 이벤트
            event_bus.transcription.started.emit(self.segment_id)
            start_time = time.time()
            
            # MLX 사용시
            if self.use_mlx:
                result = self.model.transcribe(self.segment['audio'])
                
                # 진행 상황 업데이트
                self.progress.emit(self.segment_id, 1.0)
                
            # Faster Whisper 사용시
            elif self.use_faster_whisper:
                result = self._transcribe_with_faster_whisper()
                
            # 기본 Whisper 사용시
            else:
                result = self._transcribe_with_whisper()
                
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 구성
            transcription_result = {
                'text': result.get('text', ''),
                'language': result.get('language', 'unknown'),
                'language_name': SUPPORTED_LANGUAGES.get(result.get('language', 'unknown'), 'Unknown'),
                'confidence': result.get('confidence', 0.0),
                'duration': processing_time,
                'audio_duration': self.segment.get('duration', 0),
                'segment_id': self.segment_id,
                'timestamp': time.time()
            }
            
            # 결과 전송
            self.transcription_result.emit(transcription_result)
            
            # 이벤트 발생
            event_bus.transcription.completed.emit(transcription_result)
            
            # 언어 감지 이벤트 발생
            event_bus.transcription.language_detected.emit(
                transcription_result['language'], 
                transcription_result['language_name']
            )

            # 명시적 메모리 정리
            import gc
            gc.collect()
            
        except Exception as e:
            error_msg = str(e)
            self.logger.log_error("transcription_thread", f"전사 중 오류: {error_msg}")
            self.error.emit("transcription_error", error_msg)
            event_bus.status.error.emit("transcription_error", "전사 처리 중 오류", error_msg)
        finally:
            # 명시적으로 모델 참조 해제
            self.model = None
            # 오디오 데이터 참조 해제
            if 'audio' in self.segment:
                self.segment['audio'] = None
            
    def _transcribe_with_faster_whisper(self) -> Dict:
        """Faster Whisper로 전사 수행"""
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                # float32 -> int16 변환 및 정규화
                audio_int16 = (self.segment['audio'] * 32767).astype(np.int16)
                
                # NumPy 배열을 파일로 저장
                import wave
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(1)  # 모노
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)  # 16kHz
                    wf.writeframes(audio_int16.tobytes())
                
                # 전사 수행
                segments, info = self.model.transcribe(
                    temp_file.name,
                    beam_size=5,
                    word_timestamps=False,
                    language=None,  # 자동 감지
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)  # 0.5초 이상의 묵음 제거
                )
                
                # 세그먼트별 진행 상황 업데이트를 위한 준비
                segments_list = list(segments)
                total_segments = len(segments_list)
                
                # 세그먼트 텍스트 모으기
                texts = []
                for i, segment in enumerate(segments_list):
                    texts.append(segment.text)
                    
                    # 진행 상황 업데이트
                    progress = (i + 1) / total_segments
                    self.progress.emit(self.segment_id, progress)
                
                full_text = " ".join(texts).strip()
                
                # 결과 구성
                return {
                    'text': full_text,
                    'language': info.language,
                    'confidence': info.language_probability
                }
                
        except Exception as e:
            self.logger.log_error("faster_whisper", f"Faster Whisper 전사 중 오류: {str(e)}")
            raise
            
    def _transcribe_with_whisper(self) -> Dict:
        """기본 Whisper로 전사 수행"""
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                # float32 -> int16 변환 및 정규화
                audio_int16 = (self.segment['audio'] * 32767).astype(np.int16)
                
                # NumPy 배열을 파일로 저장
                import wave
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(1)  # 모노
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)  # 16kHz
                    wf.writeframes(audio_int16.tobytes())
                
                # 전사 수행
                result = self.model.transcribe(temp_file.name)
                
                # 진행 상황 업데이트
                self.progress.emit(self.segment_id, 1.0)
                
                # 결과 구성
                return {
                    'text': result['text'],
                    'language': result['language'],
                    'confidence': 1.0  # 기본 Whisper는 신뢰도 제공 안함
                }
                
        except Exception as e:
            self.logger.log_error("whisper", f"Whisper 전사 중 오류: {str(e)}")
            raise

    def __del__(self):
        """소멸자: 리소스 정리"""
        try:
            # 스레드가 아직 실행 중이면 대기
            if self.isRunning():
                self.logger.log_debug("TranscriptionThread: 스레드 정리 중...")
                self.wait(1000)  # 최대 1초 대기
        except Exception as e:
            print(f"TranscriptionThread 소멸자 오류: {str(e)}")

    def cleanup_threads(self):
        """모든 활성 스레드를 정리"""
        if hasattr(self.transcriber, 'active_threads'):
            with self.transcriber._lock:
                active_thread_ids = list(self.transcriber.active_threads.keys())
                for thread_id in active_thread_ids:
                    thread = self.transcriber.active_threads[thread_id]
                    if thread.isRunning():
                        self.logger.log_info(f"종료 대기 중인 스레드: {thread_id}")
                        thread.wait(1000)  # 최대 1초 대기
                    # 스레드 제거
                    del self.transcriber.active_threads[thread_id]
                    
            self.logger.log_info("모든 스레드가 정리되었습니다")


class TranslationThread(QThread):
    """
    번역 작업 스레드
    - 별도 스레드에서 번역 실행
    - 결과를 시그널로 반환
    """
    # 번역 결과 시그널
    translation_result = Signal(dict)
    
    # 오류 발생 시그널
    error = Signal(str, str)
    
    def __init__(self, transcription_result: Dict, target_lang: str = "ko"):
        """번역 스레드 초기화"""
        super().__init__()
        self.logger = LogManager()
        self.transcription = transcription_result
        self.target_lang = target_lang
        self.segment_id = transcription_result.get('segment_id', 0)
        
    def run(self):
        """스레드 실행"""
        try:
            # 번역 시작 이벤트
            event_bus.transcription.translation_started.emit(self.segment_id)
            start_time = time.time()
            
            # 소스 언어 설정
            source_lang = self.transcription.get('language', 'auto')
            if source_lang == 'unknown':
                source_lang = 'auto'
                
            # 번역 대상 텍스트
            text = self.transcription.get('text', '')
            if not text:
                raise ValueError("번역할 텍스트가 없습니다")
                
            # 번역기 생성 및 번역 수행
            translator = GoogleTranslator(source=source_lang, target=self.target_lang)
            translated_text = translator.translate(text)
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 구성
            translation_result = {
                'text': translated_text,
                'source_lang': source_lang,
                'target_lang': self.target_lang,
                'duration': processing_time,
                'segment_id': self.segment_id,
                'original_text': text
            }
            
            # 결과 전송
            self.translation_result.emit(translation_result)
            
            # 이벤트 발생
            event_bus.transcription.translation_completed.emit(translation_result)
            
        except Exception as e:
            error_msg = str(e)
            self.logger.log_error("translation_thread", f"번역 중 오류: {error_msg}")
            self.error.emit("translation_error", error_msg)
            event_bus.status.error.emit("translation_error", "번역 처리 중 오류", error_msg)


class WhisperTranscriber(QObject):
    """
    Whisper 기반 음성 전사 클래스 (Qt 버전)
    - 오디오 세그먼트의 음성을 텍스트로 변환
    - 언어 감지 및 처리
    - 번역 기능 (옵션)
    """
    # 전사 완료 시그널
    transcription_completed = Signal(dict)
    
    # 번역 완료 시그널
    translation_completed = Signal(dict)
    
    # 오류 발생 시그널
    error_occurred = Signal(str, str)
    
    def __init__(self, model_name: str = "large-v3", use_faster_whisper: bool = True,
                translator_enabled: bool = True, translate_to: str = 'ko'):
        """Whisper 전사기 초기화"""
        super().__init__()
        self.logger = LogManager()
        self.logger.log_info(f"Whisper 전사기 초기화 (모델: {model_name})")
        event_bus.status.log.emit("info", f"Whisper 전사기 초기화 (모델: {model_name})")
    
        self.model_name = model_name
        self.use_faster_whisper = use_faster_whisper
        self.translator_enabled = translator_enabled
        self.translate_to = translate_to
    
        self.text_processor = TextProcessor()
    
        # 번역기 초기화 여부 확인
        self.translator_initialized = False
        if self.translator_enabled:
            self._init_translator()
    
        # Whisper 모델 초기화
        self.model = None
        self.use_mlx = False
        self._init_model()
    
        # 상태 변수
        self._lock = threading.Lock()
        self.active_threads = {}  # 활성 스레드 추적
        
        # 캐시 관리
        self.cache = {}  # 동일 오디오에 대한 결과 캐싱
        self.cache_timestamps = {}  # 각 캐시 항목의 마지막 사용 시간
        self.max_cache_size = 50  # 캐시 크기
        self.cache_memory_limit = 100 * 1024 * 1024  # 100MB 메모리 제한
        self.current_cache_memory = 0  # 현재 캐시 메모리 사용량
    
        # 통계 변수 초기화
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'avg_processing_time': 0,
            'language_counts': {},
            'success_rate': 1.0,
            'cache_evictions': 0
        }
    
        self.logger.log_info("Whisper 전사기가 초기화되었습니다")
        event_bus.status.log.emit("info", "Whisper 전사기가 초기화되었습니다")

    def _init_translator(self):
        """번역기 초기화"""
        try:
            # 번역기는 실제 사용 시점에 초기화
            self.translator_initialized = True
            self.logger.log_info("구글 번역기 사용 준비 완료")
            event_bus.status.log.emit("info", "구글 번역기 사용 준비 완료")
        except Exception as e:
            self.logger.log_error("translator_init", f"번역기 초기화 중 오류: {str(e)}")
            event_bus.status.error.emit("translator_init_error", "번역기 초기화 중 오류", str(e))
            self.translator_enabled = False

    def _init_model(self):
        """Whisper 모델 초기화"""
        try:
            # Apple Silicon 확인 - MLX 사용 시도
            if self._is_mac_silicon():
                try:
                    from mlx_whisper import MLXWhisperTranscriber
                    
                    self.logger.log_info("Apple Silicon 감지됨: LightningWhisperMLX 사용")
                    event_bus.status.log.emit("info", "Apple Silicon 감지됨: MLX 가속 활성화")
                    
                    self.model = MLXWhisperTranscriber(model_name=self.model_name)
                    self.use_mlx = True
                    return
                except Exception as e:
                    self.logger.log_warning(f"MLX 모델 초기화 실패, 대체 방법 시도: {str(e)}")
                    event_bus.status.log.emit("warning", "MLX 모델 초기화 실패, 대체 방법 시도")
                    self.use_mlx = False
            else:
                self.use_mlx = False
                
            # MLX 사용 실패 시 Faster Whisper 시도
            if self.use_faster_whisper:
                try:
                    from faster_whisper import WhisperModel
                    
                    # 장치 설정
                    compute_type = "float16"  # 기본 설정
                    
                    # 가능하면 GPU 사용
                    try:
                        import torch
                        if torch.cuda.is_available():
                            device = "cuda"
                            self.logger.log_info("CUDA 지원 GPU를 사용합니다")
                            event_bus.status.log.emit("info", "CUDA 지원 GPU를 사용합니다")
                        else:
                            device = "cpu"
                            compute_type = "float32"  # CPU에서는 float32 사용
                            self.logger.log_info("CPU를 사용합니다 (GPU 사용 불가)")
                            event_bus.status.log.emit("info", "CPU를 사용합니다 (GPU 사용 불가)")
                    except ImportError:
                        device = "cpu"
                        compute_type = "float32"  # CPU에서는 float32 사용
                        self.logger.log_info("PyTorch가 설치되지 않았습니다. CPU를 사용합니다.")
                        event_bus.status.log.emit("info", "PyTorch가 설치되지 않았습니다. CPU를 사용합니다.")
                    
                    self.logger.log_info(f"계산 타입: {compute_type}")
                    
                    # 모델 로드
                    self.model = WhisperModel(
                        self.model_name,
                        device=device,
                        compute_type=compute_type
                    )
                    self.logger.log_info(f"Faster Whisper 모델 '{self.model_name}'이 로드되었습니다 (장치: {device})")
                    event_bus.status.log.emit("info", f"Faster Whisper 모델 '{self.model_name}'이 로드되었습니다")
                    
                except ImportError:
                    self.logger.log_warning("Faster Whisper를 가져올 수 없습니다. 기본 Whisper로 대체합니다.")
                    event_bus.status.log.emit("warning", "Faster Whisper를 가져올 수 없습니다. 기본 Whisper로 대체합니다.")
                    self.use_faster_whisper = False
            
            # 기본 Whisper 사용 (다른 모든 옵션 실패 시)
            if not self.use_faster_whisper and not self.use_mlx:
                import whisper
                self.model = whisper.load_model(self.model_name)
                self.logger.log_info(f"Whisper 모델 '{self.model_name}'이 로드되었습니다")
                event_bus.status.log.emit("info", f"Whisper 모델 '{self.model_name}'이 로드되었습니다")
            
            # 모델 변경 이벤트 발생
            event_bus.transcription.model_changed.emit(self.model_name)
            
        except Exception as e:
            self.logger.log_critical(f"Whisper 모델 초기화 실패: {str(e)}")
            event_bus.status.error.emit("model_init_error", "Whisper 모델 초기화 실패", str(e))
            raise RuntimeError(f"Whisper 모델을 초기화할 수 없습니다: {str(e)}")

    def _is_mac_silicon(self) -> bool:
        """Apple Silicon Mac 여부 확인"""
        try:
            import platform
            return (platform.system() == 'Darwin' and
                   (platform.machine() == 'arm64' or 'M1' in platform.processor()))
        except:
            return False

    @Slot(dict)
    def process_audio(self, segment: Dict) -> None:
        """
        오디오 세그먼트 전사 처리
        
        Args:
            segment: 오디오 데이터 포함 세그먼트 딕셔너리
        """
        audio_data = segment.get('audio')
        segment_id = segment.get('id', 0)
    
        if audio_data is None or len(audio_data) == 0:
            self.logger.log_warning("전사 처리할 오디오 데이터가 없습니다")
            event_bus.status.log.emit("warning", "전사 처리할 오디오 데이터가 없습니다")
            return
    
        # 캐시 키 생성 (오디오 데이터의 해시)
        sample_rate = 16000  # 기본 샘플링 레이트
        sample_duration = 3  # 초 단위의 샘플링 기간
        sample_size = min(len(audio_data), int(sample_rate * sample_duration))
        
        # 균등하게 분포된 샘플 추출
        if len(audio_data) > sample_size:
            indices = np.linspace(0, len(audio_data)-1, sample_size, dtype=int)
            audio_sample = audio_data[indices]
        else:
            audio_sample = audio_data
            
        # 해시 계산
        cache_key = str(hash(audio_sample.tobytes()))
    
        # 캐시 확인
        with self._lock:
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                self.logger.log_debug("캐시에서 전사 결과를 찾았습니다")
                
                # 캐시 타임스탬프 업데이트
                self.cache_timestamps[cache_key] = time.time()
                
                # 캐시된 결과에 세그먼트 ID 업데이트
                cached_result = self.cache[cache_key].copy()
                cached_result['segment_id'] = segment_id
                
                # 결과 발생
                self.transcription_completed.emit(cached_result)
                event_bus.transcription.completed.emit(cached_result)
                
                # 번역이 있다면 번역 결과도 발생
                if 'translation' in cached_result:
                    translation = cached_result['translation'].copy()
                    translation['segment_id'] = segment_id
                    self.translation_completed.emit(translation)
                    event_bus.transcription.translation_completed.emit(translation)
                
                return
    
        # 전사 스레드 생성 및 시작
        try:
            transcription_thread = TranscriptionThread(
                segment, 
                self.model, 
                self.use_faster_whisper, 
                self.use_mlx
            )
            
            # 시그널 연결
            transcription_thread.transcription_result.connect(self._on_transcription_result)
            transcription_thread.error.connect(self._on_transcription_error)
            transcription_thread.progress.connect(self._on_transcription_progress)
            
            # 중요: finished 시그널 연결 추가
            transcription_thread.finished.connect(lambda: self._on_thread_finished(segment_id))

            # 스레드 추적
            with self._lock:
                # 혹시 이미 있는 스레드가 있으면 정리
                if segment_id in self.active_threads:
                    old_thread = self.active_threads[segment_id]
                    if old_thread.isRunning():
                        self.logger.log_warning(f"이미 실행 중인 스레드가 있습니다 (ID: {segment_id})")
                        old_thread.wait(500)  # 잠시 대기
                
                # 새 스레드 저장
                self.active_threads[segment_id] = transcription_thread
                
            # 스레드 시작
            transcription_thread.start()

        except Exception as e:
            self.logger.log_error("thread_creation", f"스레드 생성 중 오류: {str(e)}")
            event_bus.status.error.emit("thread_creation_error", "스레드 생성 중 오류", str(e))


    # 스레드 종료 처리를 위한 새 메서드 추가
    def _on_thread_finished(self, segment_id):
        """스레드 종료 처리"""
        try:
            with self._lock:
                if segment_id in self.active_threads:
                    thread = self.active_threads[segment_id]
                    # 스레드가 끝났는지 확인
                    if not thread.isRunning():
                        self.logger.log_debug(f"스레드 완료 (ID: {segment_id})")
                        # 딕셔너리에서 스레드 제거 전에 wait 호출
                        thread.wait()
                        del self.active_threads[segment_id]
        except Exception as e:
            self.logger.log_error("thread_cleanup", f"스레드 정리 중 오류: {str(e)}")

    def _on_transcription_result(self, transcription_result: Dict):
        """전사 결과 처리"""
        try:
            print("전사 결과 처리 시작")
            segment_id = transcription_result.get('segment_id', 0)
            
            # 먼저 스레드 정리 - 중요!
            with self._lock:
                if segment_id in self.active_threads:
                    # 스레드 객체 저장 (참조 유지)
                    thread = self.active_threads[segment_id]
                    # 딕셔너리에서 제거
                    del self.active_threads[segment_id]
            
            # 이후에 텍스트 처리 및 캐시 작업 수행
            processed_text = self.text_processor.process_text(transcription_result.get('text', ''))
            
            if not processed_text:
                self.logger.log_warning("후처리 후 텍스트가 비어있습니다")
                return
                    
            # 결과 업데이트
            transcription_result['text'] = processed_text
            
            # 캐시 저장
            self._cache_result(transcription_result)
            
            # 통계 업데이트
            with self._lock:
                self.stats['total_processed'] += 1
                
                # 언어별 카운트
                language = transcription_result.get('language', 'unknown')
                if language in self.stats['language_counts']:
                    self.stats['language_counts'][language] += 1
                else:
                    self.stats['language_counts'][language] = 1
                    
                # 처리 시간 통계
                self.stats['avg_processing_time'] = (
                    (self.stats['avg_processing_time'] * (self.stats['total_processed'] - 1) +
                    transcription_result.get('duration', 0)) / self.stats['total_processed']
                )
                
                # 성공률 업데이트
                self.stats['success_rate'] = 0.95 * self.stats['success_rate'] + 0.05
            
            # 결과 시그널 발생
            # self.transcription_completed.emit(transcription_result)
            event_bus.transcription.completed.emit(transcription_result)
            
            # 필요시 번역 처리
            if self.translator_enabled:
                detected_language = transcription_result.get('language', 'unknown')
                
                if detected_language != self.translate_to and detected_language in SUPPORTED_LANGUAGES:
                    self._translate_text(transcription_result)
                
        except Exception as e:
            self.logger.log_error("transcription_result", f"전사 결과 처리 중 오류: {str(e)}")
            print(f"전사 결과 처리 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            event_bus.status.error.emit("transcription_result_error", "전사 결과 처리 중 오류", str(e))

    def _translate_text(self, transcription_result: Dict):
        """번역 스레드 생성 및 시작"""
        try:
            # 번역 스레드 생성
            translation_thread = TranslationThread(transcription_result, self.translate_to)
            
            # 시그널 연결
            translation_thread.translation_result.connect(self._on_translation_result)
            translation_thread.error.connect(self._on_translation_error)
            
            # 스레드 추적
            segment_id = transcription_result.get('segment_id', 0)
            thread_key = f"translation_{segment_id}"
            
            with self._lock:
                self.active_threads[thread_key] = translation_thread
                
            # 스레드 시작
            translation_thread.start()
            
        except Exception as e:
            self.logger.log_error("translation_start", f"번역 시작 중 오류: {str(e)}")
            event_bus.status.error.emit("translation_start_error", "번역 시작 중 오류", str(e))

    @Slot(dict)
    def _on_translation_result(self, translation_result: Dict):
        """번역 결과 처리"""
        try:
            segment_id = translation_result.get('segment_id', 0)
            thread_key = f"translation_{segment_id}"
            
            # 결과 발생
            self.translation_completed.emit(translation_result)
            
            # 캐시에 번역 결과 추가
            self._add_translation_to_cache(translation_result)
            
            # 활성 스레드에서 제거
            with self._lock:
                if thread_key in self.active_threads:
                    del self.active_threads[thread_key]
                    
        except Exception as e:
            self.logger.log_error("translation_result", f"번역 결과 처리 중 오류: {str(e)}")
            event_bus.status.error.emit("translation_result_error", "번역 결과 처리 중 오류", str(e))

    @Slot(str, str)
    def _on_transcription_error(self, error_code: str, error_message: str):
        """전사 오류 처리"""
        self.error_occurred.emit(error_code, error_message)
        
        # 통계 업데이트
        with self._lock:
            self.stats['total_processed'] += 1
            # 성공률 감소
            self.stats['success_rate'] = 0.95 * self.stats['success_rate']

    @Slot(str, str)
    def _on_translation_error(self, error_code: str, error_message: str):
        """번역 오류 처리"""
        self.error_occurred.emit(error_code, error_message)

    @Slot(int, float)
    def _on_transcription_progress(self, segment_id: int, progress: float):
        """전사 진행 상황 처리"""
        # 이벤트 발생
        event_bus.transcription.progress.emit(segment_id, progress)

    def _cache_result(self, result: Dict):
        """결과 캐싱"""
        try:
            # 캐시 키 생성
            audio_data = result.get('audio', None)
            if audio_data is None:
                return
                
            # 샘플링하여 해시 생성
            sample_rate = 16000
            sample_duration = 3
            sample_size = min(len(audio_data), int(sample_rate * sample_duration))
            
            if len(audio_data) > sample_size:
                indices = np.linspace(0, len(audio_data)-1, sample_size, dtype=int)
                audio_sample = audio_data[indices]
            else:
                audio_sample = audio_data
                
            cache_key = str(hash(audio_sample.tobytes()))
            
            # 캐시 관리
            with self._lock:
                self._manage_cache()
                
                # 캐시에 결과 저장
                self.cache[cache_key] = result.copy()
                self.cache_timestamps[cache_key] = time.time()
                
                # 메모리 사용량 추적 (단순화된 추정)
                self.current_cache_memory += sys.getsizeof(str(result))
                
        except Exception as e:
            self.logger.log_error("cache", f"결과 캐싱 중 오류: {str(e)}")

    def _add_translation_to_cache(self, translation_result: Dict):
        """캐시에 번역 결과 추가"""
        try:
            segment_id = translation_result.get('segment_id', 0)
            
            # 관련 캐시 항목 찾기
            with self._lock:
                for key, cached_result in self.cache.items():
                    if cached_result.get('segment_id') == segment_id:
                        # 번역 결과 추가
                        self.cache[key]['translation'] = translation_result.copy()
                        # 타임스탬프 업데이트
                        self.cache_timestamps[key] = time.time()
                        break
                        
        except Exception as e:
            self.logger.log_error("cache_translation", f"번역 결과 캐싱 중 오류: {str(e)}")

    def _manage_cache(self):
        """캐시 크기 관리"""
        # 캐시 크기가 제한을 초과하면 오래된 항목부터 제거
        if len(self.cache) >= self.max_cache_size or self.current_cache_memory >= self.cache_memory_limit:
            # 가장 오래된 항목 찾기
            oldest_key = min(self.cache_timestamps.items(), key=lambda x: x[1])[0]
            
            # 제거할 항목의 크기 추정
            removed_size = sys.getsizeof(str(self.cache[oldest_key]))
            
            # 캐시에서 제거
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]
            
            # 메모리 사용량 업데이트
            self.current_cache_memory = max(0, self.current_cache_memory - removed_size)
            
            # 통계 업데이트
            self.stats['cache_evictions'] += 1
            
            self.logger.log_debug(f"캐시 항목 제거됨 (현재 크기: {len(self.cache)})")

    def get_stats(self) -> Dict:
        """전사기 통계 정보 반환"""
        with self._lock:
            stats_copy = self.stats.copy()
            # 캐시 사용률 추가
            stats_copy['cache_usage'] = len(self.cache)
            stats_copy['cache_hit_ratio'] = (
                stats_copy['cache_hits'] / stats_copy['total_processed']
                if stats_copy['total_processed'] > 0 else 0
            )
            return stats_copy
    
    def clear_cache(self):
        """캐시 완전 삭제"""
        with self._lock:
            self.cache.clear()
            self.cache_timestamps.clear()
            self.current_cache_memory = 0
            self.logger.log_info("캐시가 완전히 정리되었습니다")
            event_bus.status.log.emit("info", "캐시가 완전히 정리되었습니다")

    def set_translate_language(self, language_code: str) -> bool:
        """번역 대상 언어 설정"""
        if language_code in SUPPORTED_LANGUAGES:
            self.translate_to = language_code
            self.logger.log_info(f"번역 대상 언어가 변경되었습니다: {SUPPORTED_LANGUAGES[language_code]}")
            event_bus.status.log.emit("info", f"번역 대상 언어가 변경되었습니다: {SUPPORTED_LANGUAGES[language_code]}")
            return True
        self.logger.log_warning(f"지원되지 않는 언어 코드: {language_code}")
        event_bus.status.log.emit("warning", f"지원되지 않는 언어 코드: {language_code}")
        return False


class TranscriptionManager(QObject):
    """
    전사 관리 클래스 (Qt 버전)
    - 오디오 세그먼트 처리 조정
    - 결과 관리 및 필터링
    - 세션 컨텍스트 유지
    """
    # 전사 결과 시그널
    transcription_result = Signal(dict)
    
    # 번역 결과 시그널
    translation_result = Signal(dict)
    
    # 전사 세션 변경 시그널
    session_changed = Signal(str)
    
    def __init__(self, model_name: str = "large-v3", use_faster_whisper: bool = True,
                translator_enabled: bool = True, translate_to: str = 'ko'):
        """전사 관리자 초기화"""
        super().__init__()
        self.logger = LogManager()
        self.logger.log_info("전사 관리자 초기화")
        event_bus.status.log.emit("info", "전사 관리자 초기화")

        # 전사기 초기화
        self.transcriber = WhisperTranscriber(
            model_name=model_name,
            use_faster_whisper=use_faster_whisper,
            translator_enabled=translator_enabled,
            translate_to=translate_to
        )
        
        # 시그널 연결
        self.transcriber.transcription_completed.connect(self._on_transcription_completed)
        self.transcriber.translation_completed.connect(self._on_translation_completed)
        self.transcriber.error_occurred.connect(self._on_error)

        # 세션 정보
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()

        # 결과 기록
        self.transcription_history = []  # 최근 전사 결과 기록
        self.max_history = 100  # 최대 기록 수
        
        # 스레드 안전성
        self._lock = threading.Lock()

        self.logger.log_info(f"전사 세션 시작: {self.session_id}")
        event_bus.status.log.emit("info", f"전사 세션 시작: {self.session_id}")
        
        # 세션 변경 이벤트 발생
        self.session_changed.emit(self.session_id)

    @Slot(dict)
    def process_segment(self, segment: Dict):
        """
        오디오 세그먼트 처리 및 전사
        
        Args:
            segment: 오디오 세그먼트 데이터
        """
        try:
            # 전사 처리
            self.transcriber.process_audio(segment)
        except Exception as e:
            self.logger.log_error("segment_processing", f"세그먼트 처리 중 오류: {str(e)}")
            event_bus.status.error.emit("segment_processing_error", "세그먼트 처리 중 오류", str(e))

    @Slot(dict)
    def _on_transcription_completed(self, result: Dict):
        """전사 완료 처리"""
        try:
            # 깊은 복사를 사용하여 원본 객체 변경 방지
            import copy
            safe_result = copy.deepcopy(result)
            
            # 결과 기록 추가
            with self._lock:
                self.transcription_history.append(safe_result)
                # 최대 기록 수 제한
                if len(self.transcription_history) > self.max_history:
                    self.transcription_history.pop(0)
                        
            # 결과 시그널 발생 - 깊은 복사 사용
            self.transcription_result.emit(safe_result)
            
        except Exception as e:
            self.logger.log_error("transcription_result", f"전사 결과 처리 중 오류: {str(e)}")
            print(f"TranscriptionManager: 전사 결과 처리 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            event_bus.status.error.emit("transcription_result_error", "전사 결과 처리 중 오류", str(e))

    @Slot(dict)
    def _on_translation_completed(self, result: Dict):
        """번역 완료 처리"""
        try:
            # 번역 결과에 해당하는 전사 결과 업데이트
            segment_id = result.get('segment_id', 0)
            
            with self._lock:
                for item in self.transcription_history:
                    if item.get('segment_id') == segment_id:
                        item['translation'] = result.copy()
                        break
                        
            # 결과 시그널 발생
            self.translation_result.emit(result)
            
        except Exception as e:
            self.logger.log_error("translation_result", f"번역 결과 처리 중 오류: {str(e)}")
            event_bus.status.error.emit("translation_result_error", "번역 결과 처리 중 오류", str(e))

    @Slot(str, str)
    def _on_error(self, error_code: str, error_message: str):
        """오류 처리"""
        event_bus.status.error.emit(error_code, "전사/번역 중 오류", error_message)

    def get_recent_transcriptions(self, count: int = 5) -> List[Dict]:
        """최근 전사 결과 반환"""
        with self._lock:
            return self.transcription_history[-count:] if self.transcription_history else []

    def get_session_transcript(self) -> str:
        """현재 세션의 전체 전사 결과 텍스트 반환"""
        with self._lock:
            texts = [item['text'] for item in self.transcription_history if 'text' in item]
            return "\n".join(texts)

    def get_statistics(self) -> Dict:
        """전사 관리자 및 전사기 통계 반환"""
        transcriber_stats = self.transcriber.get_stats()

        stats = {
            'session_id': self.session_id,
            'session_duration': time.time() - self.session_start_time,
            'total_transcriptions': len(self.transcription_history),
            'transcriber': transcriber_stats
        }

        return stats

    def save_transcript(self, filename: str) -> bool:
        """현재 세션의 전사 결과를 파일로 저장"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # 세션 정보 헤더
                session_info = {
                    'session_id': self.session_id,
                    'start_time': datetime.datetime.fromtimestamp(self.session_start_time).strftime("%Y-%m-%d %H:%M:%S"),
                    'end_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'duration': time.time() - self.session_start_time,
                    'total_items': len(self.transcription_history)
                }

                # JSON 형식으로 저장
                json.dump({
                    'session_info': session_info,
                    'transcriptions': self.transcription_history
                }, f, ensure_ascii=False, indent=2)

            self.logger.log_info(f"전사 결과가 저장되었습니다: {filename}")
            event_bus.status.log.emit("info", f"전사 결과가 저장되었습니다: {filename}")
            return True

        except Exception as e:
            self.logger.log_error("save_transcript", f"전사 결과 저장 중 오류: {str(e)}")
            event_bus.status.error.emit("save_transcript_error", "전사 결과 저장 중 오류", str(e))
            return False

    def export_text(self, filename: str, include_translations: bool = True) -> bool:
        """텍스트 형식으로 전사 결과 내보내기"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # 헤더 정보
                f.write(f"# 전사 세션: {self.session_id}\n")
                f.write(f"# 날짜: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # 전사 내용
                for idx, item in enumerate(self.transcription_history, 1):
                    timestamp = datetime.datetime.fromtimestamp(item.get('timestamp', 0)).strftime("%H:%M:%S")
                    language = item.get('language_name', '알 수 없음')

                    f.write(f"[{idx}] {timestamp} [{language}]\n")
                    f.write(f"{item.get('text', '')}\n")

                    # 번역 포함 (옵션)
                    if include_translations and 'translation' in item:
                        f.write(f"번역: {item['translation']['text']}\n")

                    f.write("\n")

            self.logger.log_info(f"텍스트 형식으로 내보내기 완료: {filename}")
            event_bus.status.log.emit("info", f"텍스트 형식으로 내보내기 완료: {filename}")
            return True

        except Exception as e:
            self.logger.log_error("export_text", f"텍스트 내보내기 중 오류: {str(e)}")
            event_bus.status.error.emit("export_text_error", "텍스트 내보내기 중 오류", str(e))
            return False

    def reset_session(self):
        """현재 세션 초기화 (기록 삭제)"""
        with self._lock:
            self.transcription_history.clear()
            self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_start_time = time.time()

        self.logger.log_info(f"세션이 초기화되었습니다. 새 세션 ID: {self.session_id}")
        event_bus.status.log.emit("info", f"세션이 초기화되었습니다. 새 세션 ID: {self.session_id}")
        
        # 세션 변경 이벤트 발생
        self.session_changed.emit(self.session_id)