"""
메인 애플리케이션 클래스 - Qt 이벤트 시스템 사용 버전
- 전체 애플리케이션 관리
- 컴포넌트 통합 및 조정
"""

import os
import sys
import time
import argparse
import json
import importlib.util
from typing import Dict, Optional, List, Any
from datetime import datetime
from PySide6.QtCore import QObject, Signal, Slot, QThread, QTimer
from PySide6.QtWidgets import QApplication

from logging_utils import LogManager
from events import event_bus
from audio_device_qt import AudioDeviceManager, AudioRecorder
from audio_processor_qt import AudioProcessor
from transcription_qt import TranscriptionManager


class AppConfig:
    """
    애플리케이션 설정 관리 클래스
    - 기본 설정 제공
    - 설정 파일 로드/저장
    - 명령줄 인수 처리
    """
    def __init__(self, args=None):
        """기본 설정으로 초기화"""
        # 기본 설정
        self.config = {
            'model_name': 'medium',            # Whisper 모델 이름
            'use_faster_whisper': False,       # Faster Whisper 사용 여부 (기본값: 사용 안함)
            'translator_enabled': True,         # 번역 활성화 여부
            'translate_to': 'ko',                # 번역 대상 언어
            'save_transcript': True,             # 전사 결과 자동 저장 여부
            'output_dir': 'results',             # 결과 저장 디렉토리
            'log_level': 'info',                 # 로그 레벨 (debug/info)
            'calibration_duration': 3,           # 초기 환경 캘리브레이션 시간 (초)
            'ui_theme': 'light',                 # UI 테마 (light/dark)
            'subtitle_enabled': False,           # 자막 활성화 여부
            'subtitle_font_size': 24,            # 자막 폰트 크기
            'subtitle_position': 'bottom',       # 자막 위치 (top/bottom)
            'llm_api_enabled': False,            # LLM API 활성화 여부
            'auto_start': False                  # 자동 시작 여부
        }
        
        # 명령줄 인수 처리
        if args:
            self._parse_args(args)
            
        # 설정 디렉토리 생성
        self._ensure_dirs()
        
    def _ensure_dirs(self):
        """필요한 디렉토리 생성"""
        # 출력 디렉토리 생성
        if self.config['save_transcript']:
            os.makedirs(self.config['output_dir'], exist_ok=True)
            
        # 설정 디렉토리 생성
        config_dir = os.path.join(os.path.expanduser('~'), '.whisper_transcribe')
        os.makedirs(config_dir, exist_ok=True)
        
    def _parse_args(self, args):
        """명령줄 인수 처리"""
        parser = argparse.ArgumentParser(description='실시간 음성 인식 및 전사 프로그램')

        parser.add_argument('--model', type=str, default=self.config['model_name'],
                            help='사용할 Whisper 모델 (tiny, base, small, medium, large-v3 등)')

        parser.add_argument('--faster-whisper', action='store_true',
                            help='Faster Whisper 사용 (기본값: 사용 안함)')

        parser.add_argument('--no-translate', action='store_true',
                            help='자동 번역 비활성화')

        parser.add_argument('--translate-to', type=str, default=self.config['translate_to'],
                            help='번역 대상 언어 코드 (기본값: ko)')

        parser.add_argument('--no-save', action='store_true',
                            help='자동 저장 비활성화')

        parser.add_argument('--output-dir', type=str, default=self.config['output_dir'],
                            help='출력 디렉토리 경로 (기본값: results)')

        parser.add_argument('--debug', action='store_true',
                            help='디버그 로그 활성화')

        parser.add_argument('--config', type=str,
                            help='설정 파일 경로 (JSON)')
                            
        parser.add_argument('--theme', type=str, choices=['light', 'dark'], default=self.config['ui_theme'],
                            help='UI 테마 (light/dark)')
                            
        parser.add_argument('--enable-subtitle', action='store_true',
                            help='자막 활성화')
                            
        parser.add_argument('--auto-start', action='store_true',
                            help='자동 시작')

        parsed_args = parser.parse_args(args)

        # 인수를 설정 딕셔너리로 변환
        if parsed_args.model:
            self.config['model_name'] = parsed_args.model
            
        self.config['use_faster_whisper'] = parsed_args.faster_whisper
        self.config['translator_enabled'] = not parsed_args.no_translate
        self.config['translate_to'] = parsed_args.translate_to
        self.config['save_transcript'] = not parsed_args.no_save
        self.config['output_dir'] = parsed_args.output_dir
        self.config['log_level'] = 'debug' if parsed_args.debug else 'info'
        self.config['ui_theme'] = parsed_args.theme
        self.config['subtitle_enabled'] = parsed_args.enable_subtitle
        self.config['auto_start'] = parsed_args.auto_start

        # 설정 파일이 제공된 경우 로드 및 병합
        if parsed_args.config:
            self.load_config(parsed_args.config)
            
    def load_config(self, filename):
        """설정 파일 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                self.config.update(file_config)
                return True
        except Exception as e:
            print(f"설정 파일 로드 중 오류: {e}")
            return False
            
    def save_config(self, filename):
        """설정 파일 저장"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
                return True
        except Exception as e:
            print(f"설정 파일 저장 중 오류: {e}")
            return False
            
    def get(self, key, default=None):
        """설정 값 조회"""
        return self.config.get(key, default)
        
    def set(self, key, value):
        """설정 값 변경"""
        if key in self.config:
            old_value = self.config[key]
            self.config[key] = value
            
            # 이벤트 발생
            event_bus.ui.setting_changed.emit(key, value)
            
            return old_value
        return None
        
    def get_all(self):
        """모든 설정 반환"""
        return self.config.copy()


class RealTimeTranscriberApp(QObject):
    """
    실시간 음성 인식 애플리케이션 클래스
    - 전체 애플리케이션 흐름 관리
    - 모듈 간 연결 및 조정
    """
    # 초기화 완료 시그널
    initialized = Signal(bool)
    
    # 현재 상태 시그널
    status_changed = Signal(str)
    
    def __init__(self, config: Optional[Dict] = None, args=None):
        """실시간 음성 인식 앱 초기화"""
        super().__init__()
        
        # Qt 애플리케이션 객체
        self.qt_app = QApplication.instance()
        if self.qt_app is None:
            self.qt_app = QApplication(sys.argv)
        
        # 로거 초기화
        self.logger = LogManager()
        
        # 설정 객체 생성
        self.config = AppConfig(args)
        if config:
            self.config.config.update(config)
            
        # 로그 레벨 설정
        import logging
        if self.config.get('log_level') == 'debug':
            self.logger.set_log_level(logging.DEBUG)
        else:
            self.logger.set_log_level(logging.INFO)  # 명시적으로 INFO 설정

        self.logger.log_info("실시간 음성 인식기를 초기화합니다")
        event_bus.status.log.emit("info", "실시간 음성 인식기를 초기화합니다")
        
        # 상태 변수
        self.is_running = False
        self.is_processing = False
        self.device_index = None
        self.device_config = None
        
        # 컴포넌트 객체
        self.device_manager = None
        self.recorder = None
        self.processor = None
        self.transcription_manager = None
        
        # 타이머 설정
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.setInterval(5000)  # 5초마다 상태 업데이트
        
        # 이벤트 연결
        self._connect_events()
        
        # 초기화 완료 플래그
        self.is_initialized = False
        
        # 세션 정보
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.log_info("초기화가 완료되었습니다")
        event_bus.status.log.emit("info", "초기화가 완료되었습니다")
        
    def _connect_events(self):
        """이벤트 연결"""
        # 상태 이벤트 연결
        event_bus.status.updated.connect(self._on_status_updated)
        event_bus.status.error.connect(self._on_error)
        
        # 오디오 이벤트 연결
        event_bus.audio.device_initialized.connect(self._on_device_initialized)
        event_bus.audio.capture_started.connect(self._on_capture_started)
        event_bus.audio.capture_stopped.connect(self._on_capture_stopped)
        
        # 설정 변경 이벤트 연결
        event_bus.ui.setting_changed.connect(self._on_setting_changed)
        
        # 명령 이벤트 연결
        event_bus.ui.command.connect(self._on_command)

        # 전사 결과 출력 이벤트 연결 (추가)
        event_bus.transcription.completed.connect(self._on_transcription_for_log)
        event_bus.transcription.translation_completed.connect(self._on_translation_for_log)
        
    def initialize_components(self):
        """모든 구성 요소 초기화"""
        try:
            # 장치 관리자 초기화
            self.device_manager = AudioDeviceManager()
            
            # 장치 목록 시작 시 자동 로드
            self.logger.log_info("장치 목록 로드 중...")
            event_bus.status.log.emit("info", "장치 목록 로드 중...")
            devices = self.device_manager.list_devices()
            
            # 기본 장치 정보 얻기
            default_device = self.device_manager.get_default_device()
            if default_device:
                self.logger.log_info(f"기본 장치: {default_device['name']} (인덱스: {default_device['index']})")
                event_bus.status.log.emit("info", f"기본 장치: {default_device['name']}")
            
            # 초기화 성공
            self.is_initialized = True
            self.initialized.emit(True)
            self.logger.log_info("모든 구성 요소가 초기화되었습니다")
            event_bus.status.log.emit("info", "모든 구성 요소가 초기화되었습니다")
            
            # 시작 시간 표시
            self.logger.log_info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.log_info(f"세션 ID: {self.session_id}")
            
            # 상태 타이머 시작
            self.status_timer.start()
            
            # 자동 시작 설정 확인
            if self.config.get('auto_start') and default_device:
                self.select_device(default_device['index'])
                
            return True

        except Exception as e:
            self.logger.log_critical(f"구성 요소 초기화 중 오류 발생: {str(e)}")
            event_bus.status.error.emit("init_error", "초기화 중 오류 발생", str(e))
            
            self.is_initialized = False
            self.initialized.emit(False)
            return False
            
    def select_device(self, device_index: int) -> bool:
        """오디오 장치 선택 및 설정"""
        if not self.is_initialized:
            self.logger.log_warning("구성 요소가 초기화되지 않았습니다")
            event_bus.status.log.emit("warning", "구성 요소가 초기화되지 않았습니다")
            return False
            
        try:
            # 이미 실행 중이면 중지
            if self.is_running:
                self.stop()
                
            # 장치 선택
            if not self.device_manager.select_device(device_index):
                self.logger.log_warning(f"장치 {device_index} 선택 실패")
                event_bus.status.log.emit("warning", f"장치 {device_index} 선택 실패")
                return False
                
            # 장치 인덱스 저장
            self.device_index = device_index
            
            return True
            
        except Exception as e:
            self.logger.log_error("device_selection", f"장치 선택 중 오류: {str(e)}")
            event_bus.status.error.emit("device_selection_error", "장치 선택 중 오류", str(e))
            return False

    @Slot(dict)
    def _on_transcription_for_log(self, result: Dict):
        """전사 결과를 로그에 출력"""
        try:
            # 결과가 None이거나 필수 필드가 없는 경우 처리
            if result is None:
                print("결과가 None입니다")
                return
                
            # 깊은 복사로 안전하게 처리
            import copy
            safe_result = copy.deepcopy(result)
            
            language_name = safe_result.get('language_name', '알 수 없음')
            duration = safe_result.get('audio_duration', 0)
            
            # 텍스트 처리 주의
            text = safe_result.get('text', '')
            if text is None:
                text = ""
            
            # 로그 길이 제한
            if len(text) > 500:
                text = text[:497] + "..."
                
            # 안전한 문자열 포맷팅
            try:
                log_message = f"\n[전사완료][{duration:.2f}초][{language_name}] {text}"
                self.logger.log_info(log_message)
            except Exception as format_error:
                print(f"로그 메시지 포맷팅 오류: {format_error}")
                # 기본 형식으로 시도
                self.logger.log_info(f"[전사완료] 결과 처리 중 포맷 오류")
                
        except Exception as e:
            print(f"전사 결과 로깅 중 오류: {str(e)}")
            # 스택 트레이스 출력
            import traceback
            traceback.print_exc()
            
            # 오류가 있어도 프로그램은 계속 실행되도록 함
            pass

    @Slot(dict)
    def _on_translation_for_log(self, result: Dict):
        """번역 결과를 로그에 출력"""
        duration = result.get('duration', 0)
        text = result.get('text', '')
        target_lang = result.get('target_lang', 'ko')
        # 한국어인 경우 '한국어'로 표시, 다른 언어는 코드 그대로 표시
        target_lang_name = '한국어' if target_lang == 'ko' else target_lang
        self.logger.log_info(f"[번역완료][{duration:.2f}초][{target_lang_name}] {text}\n")

    def start(self) -> bool:
        """음성 인식 시작"""
        if not self.is_initialized:
            self.logger.log_warning("구성 요소가 초기화되지 않았습니다")
            event_bus.status.log.emit("warning", "구성 요소가 초기화되지 않았습니다")
            return False
            
        if self.device_index is None:
            self.logger.log_warning("장치가 선택되지 않았습니다")
            event_bus.status.log.emit("warning", "장치가 선택되지 않았습니다")
            return False
            
        if self.is_running:
            self.logger.log_warning("이미 실행 중입니다")
            return True
            
        try:
            # 오디오 설정 가져오기
            self.device_config = self.device_manager.get_config()
            
            # 레코더 초기화
            self.recorder = AudioRecorder(self.device_index, self.device_config)
            
            # 오디오 프로세서 초기화
            self.processor = AudioProcessor(sample_rate=self.device_config['sample_rate'])
            
            # 전사 관리자 초기화
            self.transcription_manager = TranscriptionManager(
                model_name=self.config.get('model_name'),
                use_faster_whisper=self.config.get('use_faster_whisper'),
                translator_enabled=self.config.get('translator_enabled'),
                translate_to=self.config.get('translate_to')
            )
            
            # 시그널 연결
            self.recorder.audio_data_ready.connect(self.processor.process_audio)
            self.processor.segment_ready.connect(self.transcription_manager.process_segment)
            
            # 레코더 시작
            if not self.recorder.start_recording():
                self.logger.log_error("recording_start", "녹음 시작 실패")
                event_bus.status.error.emit("recording_start_error", "녹음 시작 실패", "")
                return False
                
            # 상태 업데이트
            self.is_running = True
            self.status_changed.emit("running")
            self.logger.log_info("음성 인식이 시작되었습니다")
            event_bus.status.log.emit("info", "음성 인식이 시작되었습니다")
            
            return True
            
        except Exception as e:
            self.logger.log_error("start", f"시작 중 오류: {str(e)}")
            event_bus.status.error.emit("start_error", "시작 중 오류", str(e))
            
            # 정리
            self._cleanup_components()
            
            return False
            
    def stop(self) -> bool:
        """음성 인식 중지"""
        if not self.is_running:
            return True
            
        try:
            # 레코더 중지
            if self.recorder:
                self.recorder.stop_recording()
                
            # 상태 업데이트
            self.is_running = False
            self.status_changed.emit("stopped")
            self.logger.log_info("음성 인식이 중지되었습니다")
            event_bus.status.log.emit("info", "음성 인식이 중지되었습니다")
            
            # 컴포넌트 정리
            self._cleanup_components()
            
            return True
            
        except Exception as e:
            self.logger.log_error("stop", f"중지 중 오류: {str(e)}")
            event_bus.status.error.emit("stop_error", "중지 중 오류", str(e))
            return False
            
    def _cleanup_components(self):
        """컴포넌트 정리"""
        # 연결 해제
        if self.recorder and hasattr(self.recorder, 'audio_data_ready'):
            self.recorder.audio_data_ready.disconnect()
            
        if self.processor and hasattr(self.processor, 'segment_ready'):
            self.processor.segment_ready.disconnect()
            
        # 객체 정리
        self.recorder = None
        self.processor = None
        
    def save_results(self, custom_filename: str = None) -> bool:
        """현재 결과 저장"""
        if not self.transcription_manager:
            self.logger.log_warning("전사 관리자가 초기화되지 않았습니다")
            event_bus.status.log.emit("warning", "전사 관리자가 초기화되지 않았습니다")
            return False
            
        try:
            # 타임스탬프로 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = custom_filename or f"transcript_{timestamp}"
            
            # 출력 디렉토리
            output_dir = self.config.get('output_dir', 'results')
            os.makedirs(output_dir, exist_ok=True)

            # JSON 및 텍스트 파일 경로
            json_path = os.path.join(output_dir, f"{filename_base}.json")
            text_path = os.path.join(output_dir, f"{filename_base}.txt")

            # JSON 형식으로 저장
            success_json = self.transcription_manager.save_transcript(json_path)

            # 텍스트 형식으로 저장
            success_text = self.transcription_manager.export_text(text_path)

            if success_json or success_text:
                self.logger.log_info("결과가 저장되었습니다")
                event_bus.status.log.emit("info", f"결과가 저장되었습니다: {filename_base}")
                return True
            else:
                self.logger.log_warning("결과 저장 실패")
                event_bus.status.log.emit("warning", "결과 저장 실패")
                return False
                
        except Exception as e:
            self.logger.log_error("save_results", f"결과 저장 중 오류: {str(e)}")
            event_bus.status.error.emit("save_results_error", "결과 저장 중 오류", str(e))
            return False
            
    def reset_session(self) -> bool:
        """세션 초기화"""
        if not self.transcription_manager:
            self.logger.log_warning("전사 관리자가 초기화되지 않았습니다")
            event_bus.status.log.emit("warning", "전사 관리자가 초기화되지 않았습니다")
            return False
            
        try:
            # 현재 세션 저장 (선택적)
            if self.config.get('save_transcript'):
                self.save_results()
                
            # 세션 초기화
            self.transcription_manager.reset_session()
            
            if self.processor:
                self.processor.reset()
                
            # 세션 ID 업데이트
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.logger.log_info(f"세션이 초기화되었습니다. 새 세션 ID: {self.session_id}")
            event_bus.status.log.emit("info", f"세션이 초기화되었습니다")
            
            return True
            
        except Exception as e:
            self.logger.log_error("reset_session", f"세션 초기화 중 오류: {str(e)}")
            event_bus.status.error.emit("reset_session_error", "세션 초기화 중 오류", str(e))
            return False
            
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        stats = {
            'session_id': self.session_id,
            'is_running': self.is_running,
            'device_index': self.device_index,
            'app_status': 'running' if self.is_running else 'stopped'
        }
        
        # 전사 관리자 통계 추가
        if self.transcription_manager:
            transcription_stats = self.transcription_manager.get_statistics()
            stats.update({
                'transcription': transcription_stats
            })
            
        # 오디오 프로세서 통계 추가
        if self.processor:
            processor_stats = self.processor.get_stats()
            stats.update({
                'processor': processor_stats
            })
            
        # 레코더 통계 추가
        if self.recorder:
            recorder_stats = self.recorder.get_status()
            stats.update({
                'recorder': recorder_stats
            })
            
        return stats
        
    def shutdown(self):
        """애플리케이션 종료"""
        # 실행 중이면 중지
        if self.is_running:
            self.stop()
            
        # TranscriptionManager의 스레드 정리 메서드 호출
        if self.transcription_manager:
            self.transcription_manager.cleanup_threads()
            
        # 결과 자동 저장
        if self.config.get('save_transcript') and self.transcription_manager:
            self.save_results()
            
        # 타이머 중지
        if self.status_timer.isActive():
            self.status_timer.stop()
            
        self.logger.log_info("애플리케이션이 종료됩니다")
        event_bus.status.log.emit("info", "애플리케이션이 종료됩니다")
        
        # 모든 스레드가 종료될 때까지 잠시 대기 (최대 2초)
        QThread.msleep(2000)


    def get_devices(self) -> List[Dict]:
        """사용 가능한 오디오 장치 목록 반환"""
        if not self.device_manager:
            return []
            
        return self.device_manager.list_devices()
        
    def get_transcriptions(self, count: int = 5) -> List[Dict]:
        """최근 전사 결과 반환"""
        if not self.transcription_manager:
            return []
            
        return self.transcription_manager.get_recent_transcriptions(count)
        
    def get_session_transcript(self) -> str:
        """현재 세션의 전체 전사 결과 텍스트 반환"""
        if not self.transcription_manager:
            return ""
            
        return self.transcription_manager.get_session_transcript()
        
    @Slot()
    def _update_status(self):
        """주기적 상태 업데이트"""
        if not self.is_running:
            return
            
        try:
            # 통계 정보 수집
            stats = self.get_statistics()
            
            # 상태 이벤트 발생
            event_bus.status.stats_updated.emit(stats)
            
        except Exception as e:
            self.logger.log_error("status_update", f"상태 업데이트 중 오류: {str(e)}")
            
    @Slot(str, str)
    def _on_status_updated(self, status_code: str, message: str):
        """상태 업데이트 처리"""
        pass
        
    @Slot(str, str, str)
    def _on_error(self, error_code: str, error_message: str, details: str):
        """오류 처리"""
        self.logger.log_error(error_code, f"{error_message}: {details}")
        
    @Slot(int, dict)
    def _on_device_initialized(self, device_index: int, config: Dict):
        """장치 초기화 완료 처리"""
        self.device_index = device_index
        self.device_config = config
        
    @Slot()
    def _on_capture_started(self):
        """녹음 시작 처리"""
        self.is_running = True
        self.status_changed.emit("running")
        
    @Slot()
    def _on_capture_stopped(self):
        """녹음 중지 처리"""
        self.is_running = False
        self.status_changed.emit("stopped")
        
    @Slot(str, object)
    def _on_setting_changed(self, key: str, value: Any):
        """설정 변경 처리"""
        # 설정에 따른 동작 변경
        if key == 'model_name' and self.transcription_manager:
            self.logger.log_info(f"모델 변경: {value}")
            # 모델 변경은 재시작 필요
            
        elif key == 'translate_to' and self.transcription_manager:
            self.transcription_manager.transcriber.set_translate_language(value)
            
        elif key == 'translator_enabled' and self.transcription_manager:
            # 번역 활성화/비활성화는 재시작 필요
            self.logger.log_info(f"번역 {'활성화' if value else '비활성화'}")
            
    @Slot(str, dict)
    def _on_command(self, command: str, params: Dict):
        """명령 처리"""
        if command == 'start':
            self.start()
            
        elif command == 'stop':
            self.stop()
            
        elif command == 'save':
            filename = params.get('filename')
            self.save_results(filename)
            
        elif command == 'reset':
            confirm = params.get('confirm', False)
            if confirm:
                self.reset_session()
                
        elif command == 'select_device':
            device_index = params.get('device_index')
            if device_index is not None:
                self.select_device(device_index)


def main():
    """메인 함수 - GUI 없이 테스트용"""
    import sys
    import signal
    import threading
    
    # Qt 애플리케이션 생성
    qt_app = QApplication(sys.argv)
    
    # 앱 인스턴스 생성
    app = RealTimeTranscriberApp(args=sys.argv[1:])
    app.initialize_components()
    
    # 시그널 핸들러 설정
    def signal_handler(sig, frame):
        print("\n프로그램을 종료합니다...")
        app.shutdown()
        qt_app.quit()
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # 기본 장치 선택 - 자동으로 사용자에게 선택 요청
    default_device = app.device_manager.get_default_device()
    if default_device:
        print(f"\n기본 장치 '{default_device['name']}'를 사용합니다.")
        print("다른 장치를 사용하려면 GUI 버전을 실행하세요.")
        app.select_device(default_device['index'])
        app.start()
    else:
        print("\n기본 장치를 찾을 수 없습니다. GUI 버전을 실행하세요.")
    
    # CLI 명령어 처리 스레드 생성
    def process_commands():
        print("\n실시간 음성 인식이 시작되었습니다. 명령을 입력하려면 Enter를 누르세요.")
        print("사용 가능한 명령: stats, save, reset, config, help, exit")
        
        while True:
            try:
                # 사용자가 Enter를 누를 때까지 대기
                user_input = input("")
                
                if not user_input:
                    print("\n명령어를 입력하세요 (help로 도움말 확인):")
                    user_input = input("> ").strip().lower()
                
                if not user_input:
                    continue
                    
                # 명령 처리
                if user_input in ('q', 'quit', 'exit'):
                    print("프로그램을 종료합니다...")
                    app.shutdown()
                    qt_app.quit()
                    break
                    
                elif user_input in ('h', 'help'):
                    show_help()
                    
                elif user_input in ('s', 'stats'):
                    show_stats(app)
                    
                elif user_input in ('save', 'export'):
                    app.save_results()
                    
                elif user_input in ('r', 'reset'):
                    confirm = input("\n정말로 현재 세션을 초기화하시겠습니까? 모든 전사 기록이 삭제됩니다. (y/n): ")
                    if confirm.lower() == 'y':
                        app.reset_session()
                    
                elif user_input in ('c', 'config'):
                    show_config(app)
                    
                elif user_input.startswith('set '):
                    change_config(app, user_input[4:])
                    
                else:
                    print(f"알 수 없는 명령: {user_input}")
                    print("'help'를 입력하면 사용 가능한 명령어를 확인할 수 있습니다.")
                    
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다...")
                app.shutdown()
                qt_app.quit()
                break
                
            except Exception as e:
                print(f"\n[오류] 명령 처리 중 오류 발생: {e}")
    
    # 도움말 표시 함수
    def show_help():
        print("\n=== 사용 가능한 명령어 ===")
        print("help, h      : 이 도움말 표시")
        print("stats, s     : 현재 전사 통계 표시")
        print("save, export : 현재까지의 전사 결과 저장")
        print("reset, r     : 세션 초기화 (기록 삭제)")
        print("config, c    : 현재 설정 확인")
        print("set [옵션]   : 설정 변경 (예: set translate_to=en)")
        print("exit, quit, q: 프로그램 종료")
    
    # 통계 표시 함수
    def show_stats(app):
        stats = app.get_statistics()
        
        print("\n=== 전사 통계 ===")
        print(f"세션 ID: {stats.get('session_id', 'N/A')}")
        print(f"세션 시간: {stats.get('session_duration', 0):.1f}초")
        print(f"전사 항목 수: {stats.get('total_transcriptions', 0)}")
        
        # Whisper 통계
        if 'transcription' in stats and 'transcriber' in stats['transcription']:
            whisper_stats = stats['transcription']['transcriber']
            if whisper_stats.get('total_processed', 0) > 0:
                print(f"\n=== Whisper 통계 ===")
                print(f"처리된 세그먼트: {whisper_stats.get('total_processed', 0)}")
                print(f"평균 처리 시간: {whisper_stats.get('avg_processing_time', 0):.2f}초")
                print(f"성공률: {whisper_stats.get('success_rate', 0)*100:.1f}%")
                
                # 언어별 통계
                if 'language_counts' in whisper_stats and whisper_stats['language_counts']:
                    print("\n언어별 통계:")
                    for lang, count in whisper_stats['language_counts'].items():
                        lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
                        percentage = count / whisper_stats['total_processed'] * 100
                        print(f"  {lang_name}: {count}개 ({percentage:.1f}%)")
    
    # 설정 표시 함수
    def show_config(app):
        print("\n=== 현재 설정 ===")
        config = app.config.get_all()
        for key, value in config.items():
            print(f"{key}: {value}")
    
    # 설정 변경 함수
    def change_config(app, config_str):
        try:
            # 입력 형식: key=value
            if '=' not in config_str:
                print("잘못된 형식입니다. 'set key=value' 형식으로 입력하세요.")
                return
                
            key, value = config_str.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # 설정 변경
            old_value = app.config.set(key, value)
            if old_value is not None:
                print(f"설정이 변경되었습니다: {key} = {value}")
            else:
                print(f"알 수 없는 설정 키: {key}")
                
        except Exception as e:
            print(f"설정 변경 중 오류가 발생했습니다: {e}")
    
    # 명령어 처리 스레드 시작
    cmd_thread = threading.Thread(target=process_commands, daemon=True)
    cmd_thread.start()
    
    # 타이머를 사용하여 주기적으로 시그널 처리 (SIGINT 캡처용)
    timer = QTimer()
    timer.timeout.connect(lambda: None)  # 더미 콜백
    timer.start(100)  # 100ms마다 실행
    
    # 디버깅을 위한 예외 처리 추가
    try:
        sys.exit(qt_app.exec())
    except Exception as e:
        print(f"프로그램 종료 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 활성 스레드 목록 출력
        print("활성 스레드 목록:")
        for thread in QThreadPool.globalInstance().findChildren(QThread):
            print(f" - {thread.objectName()}: 실행 중 = {thread.isRunning()}")


if __name__ == "__main__":
    main()