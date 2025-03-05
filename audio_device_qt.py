"""
오디오 장치 관리 클래스 - Qt 이벤트 시스템 사용 버전
- 기존 audio_device.py의 기능을 이벤트 드리븐 방식으로 리팩토링
"""

import numpy as np
import pyaudio
import wave
import os
import time
import threading
from typing import Optional, List, Dict, Tuple, Any
from PySide6.QtCore import QObject, QThread, Signal, Slot

from logging_utils import LogManager
from events import event_bus


class AudioDeviceManager(QObject):
    """
    오디오 장치 관리 클래스 (Qt 버전)
    - 사용 가능한 오디오 장치 검색
    - 장치 선택 및 검증
    - 오디오 설정 구성
    """
    
    def __init__(self):
        """오디오 장치 관리자 초기화"""
        super().__init__()
        self.logger = LogManager()
        self.logger.log_info("오디오 장치 관리자를 초기화합니다")
        event_bus.status.log.emit("info", "오디오 장치 관리자를 초기화합니다")

        try:
            self.audio = pyaudio.PyAudio()
            self._device_info = None
            self._sample_rate = 16000  # Whisper 최적 샘플레이트
            self._channels = 1         # 모노 채널
            self._format = pyaudio.paFloat32  # 32비트 부동 소수점 형식
            self._chunk_size = int(self._sample_rate * 0.03)  # 30ms 청크
        except Exception as e:
            self.logger.log_critical(f"PyAudio 초기화 실패: {str(e)}")
            event_bus.status.error.emit("init_error", "PyAudio 초기화 실패", str(e))
            raise

    def list_devices(self) -> List[Dict]:
        """사용 가능한 모든 오디오 장치 목록 반환"""
        devices = []
        try:
            info = self.audio.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')

            self.logger.log_info(f"사용 가능한 장치 수: {num_devices}")
            event_bus.status.log.emit("info", f"사용 가능한 장치 수: {num_devices}")

            for i in range(num_devices):
                try:
                    device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
                    device_name = device_info.get('name')
                    max_input_channels = device_info.get('maxInputChannels')

                    # 입력 채널이 있는 장치만 추가
                    if max_input_channels > 0:
                        # Loopback 장치 확인 (시스템 오디오 캡처용)
                        is_loopback = 'Loopback' in device_name or 'loopback' in device_name.lower() or '루프백' in device_name

                        device_data = {
                            'index': i,
                            'name': device_name,
                            'channels': max_input_channels,
                            'is_loopback': is_loopback,
                            'default_sample_rate': int(device_info.get('defaultSampleRate'))
                        }

                        devices.append(device_data)
                        self.logger.log_debug(f"장치 {i} 정보: {device_data}")

                except Exception as e:
                    self.logger.log_error("device_info", f"장치 {i} 정보 조회 중 오류: {str(e)}")

            # 이벤트 발생: 장치 목록 업데이트
            event_bus.audio.devices_listed.emit(devices)

        except Exception as e:
            self.logger.log_error("list_devices", f"장치 목록 조회 중 오류: {str(e)}")
            event_bus.status.error.emit("device_list_error", "장치 목록 조회 중 오류", str(e))

        return devices

    def validate_device(self, device_index: int) -> bool:
        """선택한 장치가 유효한지 확인"""
        try:
            device_info = self.audio.get_device_info_by_host_api_device_index(0, device_index)
            if not device_info:
                self.logger.log_warning(f"장치 인덱스 {device_index}에 대한 정보를 찾을 수 없습니다")
                event_bus.status.log.emit("warning", f"장치 인덱스 {device_index}에 대한 정보를 찾을 수 없습니다")
                return False

            # 기본 검증
            if device_info.get('maxInputChannels', 0) <= 0:
                self.logger.log_warning(f"장치 {device_index}는 입력 채널이 없습니다")
                event_bus.status.log.emit("warning", f"장치 {device_index}는 입력 채널이 없습니다")
                return False

            # 샘플레이트 호환성 확인
            default_rate = int(device_info.get('defaultSampleRate', 44100))
            if default_rate < 16000:
                self.logger.log_warning(f"장치의 기본 샘플레이트({default_rate}Hz)가 권장값(16000Hz)보다 낮습니다")
                event_bus.status.log.emit("warning", f"장치의 기본 샘플레이트({default_rate}Hz)가 권장값(16000Hz)보다 낮습니다")
                # 낮은 샘플레이트도 사용은 가능하므로 경고만 출력

            # 테스트 스트림 생성
            try:
                test_stream = self.audio.open(
                    format=self._format,
                    channels=1,  # 모노로 강제
                    rate=16000,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self._chunk_size,
                    start=False
                )
                test_stream.close()
                self._device_info = device_info
                self.logger.log_info(f"장치 {device_index} 검증 성공")
                event_bus.status.log.emit("info", f"장치 {device_index} 검증 성공")
                return True
            except Exception as e:
                self.logger.log_error("device_test", f"장치 테스트 실패: {str(e)}")
                event_bus.status.error.emit("device_test_error", "장치 테스트 실패", str(e))
                return False

        except Exception as e:
            self.logger.log_error("device_validation", f"장치 검증 중 오류 발생: {str(e)}")
            event_bus.status.error.emit("device_validation_error", "장치 검증 중 오류 발생", str(e))
            return False

    def select_device(self, device_index: int) -> bool:
        """장치 선택 및 구성"""
        if self.validate_device(device_index):
            self._configure_device(device_index)
            self.logger.log_info(f"장치 {device_index} 선택됨")
            
            # 이벤트 발생: 장치 초기화 완료
            config = self.get_config()
            event_bus.audio.device_initialized.emit(device_index, config)
            return True
        return False

    def _configure_device(self, device_index: int):
        """선택된 장치에 대한 최적 설정 구성"""
        device_info = self._device_info

        # 샘플레이트 설정
        default_rate = int(device_info.get('defaultSampleRate', 44100))
        if default_rate >= 16000:
            self._sample_rate = 16000  # Whisper 모델 최적 샘플레이트
        else:
            self._sample_rate = default_rate
            self.logger.log_warning(f"최적 샘플레이트보다 낮은 값({default_rate}Hz)을 사용합니다")

        # 채널 설정
        self._channels = 1  # 모노로 강제

        # 청크 크기 조정
        self._chunk_size = int(self._sample_rate * 0.03)  # 30ms 기준

        self.logger.log_info(f"장치 구성 완료 - 샘플레이트: {self._sample_rate}Hz, 채널: {self._channels}, 청크 크기: {self._chunk_size}")
        event_bus.status.log.emit("info", f"장치 구성 완료 - 샘플레이트: {self._sample_rate}Hz, 채널: {self._channels}")

    def get_config(self) -> Dict:
        """현재 오디오 설정 반환"""
        return {
            'sample_rate': self._sample_rate,
            'channels': self._channels,
            'format': self._format,
            'chunk_size': self._chunk_size
        }

    def get_default_device(self) -> Optional[Dict]:
        """기본 입력 장치 정보 반환"""
        try:
            default_input = self.audio.get_default_input_device_info()
            if default_input:
                return {
                    'index': default_input.get('index'),
                    'name': default_input.get('name'),
                    'channels': default_input.get('maxInputChannels'),
                    'default_sample_rate': int(default_input.get('defaultSampleRate'))
                }
        except Exception as e:
            self.logger.log_debug(f"기본 장치 정보 가져오기 실패: {str(e)}")
        return None

    def __del__(self):
        """소멸자: 리소스 정리"""
        try:
            if hasattr(self, 'audio'):
                self.audio.terminate()
                self.logger.log_info("오디오 장치 리소스가 정리되었습니다")
        except Exception as e:
            self.logger.log_error("cleanup", f"리소스 정리 중 오류 발생: {str(e)}")

class AudioRecorderThread(QThread):
    """
    오디오 녹음 스레드 클래스
    - QThread를 상속받아 별도 스레드에서 오디오 녹음 처리
    """
    # 오디오 데이터 처리 시그널 (chunk_dict)
    audio_data = Signal(dict)
    
    # 오디오 레벨 업데이트 시그널 (energy_level)
    level_updated = Signal(float)
    
    # 오류 발생 시그널
    error = Signal(str, str)
    
    def __init__(self, device_index: int, config: Dict):
        """오디오 레코더 스레드 초기화"""
        super().__init__()
        self.logger = LogManager()
        self.logger.log_info(f"오디오 레코더 스레드 초기화 (장치 인덱스: {device_index})")
        
        # 설정 저장
        self.device_index = device_index
        self.config = config
        
        # 상태 변수
        self.is_recording = False
        self.stop_requested = False
        
        # 청크 크기 사용
        self.chunk_size = config.get('chunk_size', 1024)
        self.sample_rate = config.get('sample_rate', 16000)
        self.channels = config.get('channels', 1)
        self.format = config.get('format', pyaudio.paFloat32)
        
        # 스트림 상태 모니터링
        self.stream_status = {
            'overflows': 0,
            'chunks_processed': 0,
            'last_timestamp': time.time(),
            'avg_processing_time': 0
        }
        
        # 에너지 임계값 설정
        self.energy_threshold = 0.01  # RMS 에너지 임계값 (0-1 범위)
        
        # PyAudio 인스턴스
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def run(self):
        """스레드 실행 메서드"""
        try:
            self.logger.log_info("오디오 레코더 스레드 시작")
            event_bus.status.log.emit("info", "오디오 녹음 시작")
            
            self.is_recording = True
            self.stop_requested = False
            
            # 오디오 스트림 생성
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=None
            )
            
            # 이벤트 발생: 녹음 시작
            event_bus.audio.capture_started.emit()
            
            # 녹음 루프
            while not self.stop_requested:
                try:
                    # 오디오 데이터 읽기
                    process_start = time.time()
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # float32로 변환
                    if self.format == pyaudio.paInt16:
                        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    else:  # 이미 float32인 경우
                        audio_chunk = np.frombuffer(data, dtype=np.float32)
                    
                    # 간단한 전처리
                    processed_chunk = self._preprocess_audio(audio_chunk)
                    
                    # 에너지 레벨 계산
                    energy = self._calculate_energy(processed_chunk)
                    
                    # 에너지 레벨 업데이트 신호 발생
                    self.level_updated.emit(energy)
                    event_bus.audio.level_updated.emit(energy)
                    
                    # 오디오 데이터 신호 발생
                    chunk_dict = {
                        'audio': processed_chunk,
                        'energy': energy,
                        'timestamp': time.time()
                    }
                    self.audio_data.emit(chunk_dict)
                    
                    # 상태 업데이트
                    process_time = time.time() - process_start
                    self._update_status(process_time)
                    
                except IOError as e:
                    self.stream_status['overflows'] += 1
                    self.logger.log_warning(f"버퍼 오버플로우 발생 ({self.stream_status['overflows']}번째)")
                    continue
                    
                except Exception as e:
                    self.logger.log_error("recording", f"녹음 중 예외 발생: {str(e)}")
                    self.error.emit("recording_error", str(e))
                    break
                    
        except Exception as e:
            self.logger.log_error("stream", f"스트림 생성 중 예외 발생: {str(e)}")
            self.error.emit("stream_error", str(e))
            
        finally:
            # 스트림 정리
            self._cleanup()
            self.is_recording = False
            
            # 이벤트 발생: 녹음 중지
            event_bus.audio.capture_stopped.emit()
            
            self.logger.log_info("오디오 레코더 스레드 종료")
            event_bus.status.log.emit("info", "오디오 녹음 중지")

    def stop(self):
        """녹음 중지 요청"""
        self.stop_requested = True
        self.logger.log_info("녹음 중지 요청")

    def _preprocess_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        """오디오 데이터 전처리"""
        try:
            # DC offset 제거 (평균값 제거)
            audio_chunk = audio_chunk - np.mean(audio_chunk)
            
            # 클리핑 방지를 위한 정규화 (과도한 볼륨 줄이기)
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0.95:  # 클리핑 임계값
                audio_chunk = audio_chunk * (0.95 / max_val)
                
            return audio_chunk
            
        except Exception as e:
            self.logger.log_error("preprocessing", f"전처리 중 오류: {str(e)}")
            return audio_chunk  # 오류 시 원본 반환

    def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
        """오디오 에너지 레벨 계산"""
        try:
            if len(audio_chunk) == 0:
                return 0.0
            # RMS(Root Mean Square) 에너지 계산
            return np.sqrt(np.mean(np.square(audio_chunk)))
        except Exception as e:
            self.logger.log_error("energy_calculation", f"에너지 계산 중 오류: {str(e)}")
            return 0.0

    def _update_status(self, process_time: float):
        """스트림 상태 업데이트"""
        try:
            self.stream_status['chunks_processed'] += 1
            
            # 평균 처리 시간 업데이트
            alpha = 0.1  # 평활화 계수
            self.stream_status['avg_processing_time'] = (
                (1 - alpha) * self.stream_status['avg_processing_time'] +
                alpha * process_time
            )
            
            # 주기적 상태 출력 (100 청크마다)
            if self.stream_status['chunks_processed'] % 100 == 0:
                self.logger.log_debug(
                    f"처리 상태 - 청크: {self.stream_status['chunks_processed']}, "
                    f"평균 처리 시간: {self.stream_status['avg_processing_time']*1000:.1f}ms"
                )
                
                # 상태 이벤트 발생
                status_info = {
                    'chunks_processed': self.stream_status['chunks_processed'],
                    'avg_processing_time': self.stream_status['avg_processing_time'],
                    'overflows': self.stream_status['overflows']
                }
                event_bus.status.stats_updated.emit(status_info)
                
        except Exception as e:
            self.logger.log_error("status_update", f"상태 업데이트 중 오류: {str(e)}")

    def _cleanup(self):
        """리소스 정리"""
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                self.logger.log_info("오디오 스트림이 정리되었습니다")
        except Exception as e:
            self.logger.log_error("cleanup", f"스트림 정리 중 오류: {str(e)}")
            
    def get_status(self) -> Dict:
        """현재 상태 정보 반환"""
        return self.stream_status.copy()

    def __del__(self):
        """소멸자: 리소스 정리"""
        self._cleanup()
        try:
            if hasattr(self, 'audio') and self.audio is not None:
                self.audio.terminate()
                self.logger.log_info("오디오 장치 리소스가 정리되었습니다")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.log_error("cleanup", f"리소스 정리 중 오류 발생: {str(e)}")


class AudioRecorder(QObject):
    """
    오디오 녹음 클래스 (Qt 버전)
    - 스레드 관리
    - 오디오 청크 처리
    - 기본적인 전처리
    """
    # 오디오 데이터 신호 (chunk_dict)
    audio_data_ready = Signal(dict)
    
    # 오류 발생 신호
    error_occurred = Signal(str, str)
    
    def __init__(self, device_index: int, config: Dict):
        """오디오 레코더 초기화"""
        super().__init__()
        self.logger = LogManager()
        self.logger.log_info(f"오디오 레코더 초기화 (장치 인덱스: {device_index})")
        event_bus.status.log.emit("info", f"오디오 레코더 초기화 (장치: {device_index})")
        
        self.device_index = device_index
        self.config = config
        
        # 레코더 스레드
        self.recorder_thread = None
        self.is_recording = False

    def start_recording(self):
        """녹음 시작"""
        if self.is_recording:
            self.logger.log_warning("이미 녹음 중입니다")
            return False
            
        try:
            # 스레드 생성 및 시작
            self.recorder_thread = AudioRecorderThread(self.device_index, self.config)
            
            # 시그널 연결
            self.recorder_thread.audio_data.connect(self._on_audio_data)
            self.recorder_thread.level_updated.connect(self._on_level_updated)
            self.recorder_thread.error.connect(self._on_error)
            
            # 스레드 시작
            self.recorder_thread.start()
            self.is_recording = True
            
            self.logger.log_info("녹음이 시작되었습니다")
            return True
            
        except Exception as e:
            self.logger.log_error("recording_start", f"녹음 시작 중 오류: {str(e)}")
            event_bus.status.error.emit("recording_start_error", "녹음 시작 중 오류", str(e))
            return False

    def stop_recording(self):
        """녹음 중지"""
        if not self.is_recording:
            return
            
        try:
            if self.recorder_thread and self.recorder_thread.isRunning():
                self.recorder_thread.stop()
                self.recorder_thread.wait(2000)  # 최대 2초 대기
                
                # 여전히 실행 중이면 강제 종료
                if self.recorder_thread.isRunning():
                    self.logger.log_warning("녹음 스레드가 응답하지 않아 강제 종료합니다")
                    self.recorder_thread.terminate()
                
                self.recorder_thread = None
                
            self.is_recording = False
            self.logger.log_info("녹음이 중지되었습니다")
            
        except Exception as e:
            self.logger.log_error("recording_stop", f"녹음 중지 중 오류: {str(e)}")
            event_bus.status.error.emit("recording_stop_error", "녹음 중지 중 오류", str(e))

    def record_to_file(self, filename: str, duration: float) -> bool:
        """지정된 시간 동안 녹음하여 파일로 저장"""
        try:
            # 녹음 중인 경우 중지
            if self.is_recording:
                self.stop_recording()
                
            self.logger.log_info(f"{duration}초 동안 녹음을 시작합니다 - 파일: {filename}")
            event_bus.status.log.emit("info", f"{duration}초 동안 녹음을 시작합니다")
            
            # PyAudio 객체 및 스트림 생성
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=self.config['format'],
                channels=self.config['channels'],
                rate=self.config['sample_rate'],
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.config['chunk_size']
            )
            
            frames = []
            chunk_count = int(self.config['sample_rate'] * duration / self.config['chunk_size'])
            
            for i in range(chunk_count):
                # 진행 상황 업데이트 (10% 단위)
                if i % (chunk_count // 10) == 0:
                    progress = i / chunk_count
                    event_bus.status.updated.emit("recording_progress", f"{progress:.0%}")
                
                data = stream.read(self.config['chunk_size'], exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # WAV 파일로 저장
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.config['channels'])
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # WAV는 항상 Int16
            wf.setframerate(self.config['sample_rate'])
            
            # Float32 -> Int16 변환 (필요한 경우)
            if self.config['format'] == pyaudio.paFloat32:
                audio_array = np.array([])
                for frame in frames:
                    chunk = np.frombuffer(frame, dtype=np.float32)
                    audio_array = np.append(audio_array, chunk)
                
                # 클리핑 방지 정규화
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:  # 0으로 나누기 방지
                    audio_array = audio_array / max_val
                
                # Float32 -> Int16 변환
                audio_array = (audio_array * 32767).astype(np.int16)
                wf.writeframes(audio_array.tobytes())
            else:
                # 이미 Int16인 경우
                for frame in frames:
                    wf.writeframes(frame)
            
            wf.close()
            audio.terminate()
            
            self.logger.log_info(f"녹음 완료: {filename}")
            event_bus.status.log.emit("info", f"녹음 완료: {filename}")
            return True
            
        except Exception as e:
            self.logger.log_error("file_recording", f"파일 녹음 중 오류 발생: {str(e)}")
            event_bus.status.error.emit("file_recording_error", "파일 녹음 중 오류", str(e))
            return False

    @Slot(dict)
    def _on_audio_data(self, audio_data: Dict):
        """오디오 데이터 수신 처리"""
        # 오디오 데이터 시그널 릴레이
        self.audio_data_ready.emit(audio_data)

    @Slot(float)
    def _on_level_updated(self, level: float):
        """오디오 레벨 업데이트 처리"""
        # 필요한 경우 추가 처리 가능
        pass

    @Slot(str, str)
    def _on_error(self, error_code: str, error_message: str):
        """오류 수신 처리"""
        self.error_occurred.emit(error_code, error_message)
        event_bus.status.error.emit(error_code, "녹음 중 오류 발생", error_message)
        
        # 오류 발생 시 녹음 중지
        if self.is_recording:
            self.is_recording = False
            self.logger.log_warning("오류로 인해 녹음을 중지합니다")

    def get_status(self) -> Dict:
        """레코더 상태 정보 반환"""
        status = {
            'is_recording': self.is_recording,
            'device_index': self.device_index,
            'sample_rate': self.config.get('sample_rate', 0),
        }
        
        # 스레드 상태 추가
        if self.recorder_thread and self.recorder_thread.isRunning():
            status.update(self.recorder_thread.get_status())
            
        return status

    def __del__(self):
        """소멸자: 리소스 정리"""
        self.stop_recording()