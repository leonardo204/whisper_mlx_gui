"""
이벤트 시스템 모듈
- PySide6/PyQt5 시그널을 활용한 이벤트 시스템 구현
- 각 모듈 간 느슨한 결합을 위한 이벤트 정의
"""

from PySide6.QtCore import QObject, Signal, Slot
from typing import Dict, List, Optional, Any
import numpy as np
import time


class AudioEvents(QObject):
    """오디오 관련 이벤트"""
    
    # 오디오 장치 초기화 완료 (장치 인덱스, 장치 정보)
    device_initialized = Signal(int, dict)
    
    # 오디오 캡처 시작/중지
    capture_started = Signal()
    capture_stopped = Signal()
    
    # 오디오 세그먼트 감지 (세그먼트 정보)
    segment_detected = Signal(dict)
    
    # 오디오 레벨 업데이트 (에너지 레벨 0.0-1.0)
    level_updated = Signal(float)
    
    # 장치 목록 업데이트 (장치 목록)
    devices_listed = Signal(list)


class TranscriptionEvents(QObject):
    """전사 관련 이벤트"""
    
    # 전사 시작/진행/완료 (세그먼트 ID)
    started = Signal(int)
    progress = Signal(int, float)  # (세그먼트 ID, 진행률 0.0-1.0)
    completed = Signal(dict)  # 전사 결과
    
    # 번역 시작/완료
    translation_started = Signal(int)  # 세그먼트 ID
    translation_completed = Signal(dict)  # 번역 결과
    
    # 언어 감지 (감지된 언어 코드)
    language_detected = Signal(str, str)  # (언어 코드, 언어 이름)
    
    # 모델 변경 (모델 이름)
    model_changed = Signal(str)

    def emit_completed(self, result: Dict):
        """전사 완료 이벤트 안전하게 발생"""
        try:
            if not result:
                print("전사 결과가 비어있습니다")
                return
                
            # 텍스트 길이 제한
            if 'text' in result and isinstance(result['text'], str) and len(result['text']) > 500:
                result = result.copy()  # 얕은 복사로 충분함
                result['text'] = result['text'][:500] + "..."
                
            # 이벤트 발생
            self.completed.emit(result)
        except Exception as e:
            print(f"전사 완료 이벤트 발생 중 오류: {str(e)}")

class StatusEvents(QObject):
    """상태 및 로그 관련 이벤트"""
    
    # 상태 업데이트 (상태 코드, 상태 메시지)
    updated = Signal(str, str)
    
    # 오류 발생 (오류 코드, 오류 메시지, 오류 상세)
    error = Signal(str, str, str)
    
    # 로그 메시지 (로그 레벨, 메시지)
    log = Signal(str, str)
    
    # 통계 업데이트 (통계 정보 딕셔너리)
    stats_updated = Signal(dict)


class UIEvents(QObject):
    """UI 관련 이벤트"""
    
    # 사용자 설정 변경 (설정 키, 설정 값)
    setting_changed = Signal(str, object)
    
    # 사용자 명령 (명령 이름, 매개변수)
    command = Signal(str, dict)
    
    # 자막 표시/숨김 (텍스트, 지속 시간)
    subtitle_show = Signal(str, int)
    subtitle_hide = Signal()
    
    # 화면 전환 (화면 이름)
    screen_changed = Signal(str)


class EventBus(QObject):
    """이벤트 버스 - 모든 이벤트의 중앙 허브"""
    
    def __init__(self):
        super().__init__()
        
        # 이벤트 그룹 초기화
        self.audio = AudioEvents()
        self.transcription = TranscriptionEvents()
        self.status = StatusEvents()
        self.ui = UIEvents()
        
    def emit_log(self, level: str, message: str):
        """로그 이벤트 간편 발생 메서드"""
        try:
            if hasattr(self, 'status') and hasattr(self.status, 'log'):
                self.status.log.emit(level, message)
        except Exception as e:
            print(f"로그 이벤트 발생 중 오류: {str(e)}")
        
    def emit_error(self, code: str, message: str, detail: str = ""):
        """오류 이벤트 간편 발생 메서드"""
        try:
            if hasattr(self, 'status') and hasattr(self.status, 'error'):
                self.status.error.emit(code, message, detail)
        except Exception as e:
            print(f"오류 이벤트 발생 중 오류: {str(e)}")


# 전역 이벤트 버스 인스턴스
event_bus = EventBus()