import rclpy
from rclpy.node import Node
from n2r_topic_interfaces.msg import AudioText
import threading
import time
import pyaudio
import wave
import tempfile
import os
import numpy as np

class ImprovedWhisperNode(Node):
    def __init__(self):
        super().__init__('improved_whisper_node')
        
        # 创建发布者
        self.text_publisher = self.create_publisher(AudioText, 'audio_text', 10)
        
        # 参数配置
        self.declare_parameter('model_size', 'base')
        self.declare_parameter('language', 'zh')
        self.declare_parameter('record_seconds', 5)
        self.declare_parameter('silence_threshold', 500)  # 静音阈值
        self.declare_parameter('min_audio_level', 10)   # 最小音频级别
        self.declare_parameter('consecutive_silence', 10) # 连续静音帧数
        
        model_size = self.get_parameter('model_size').value
        self.language = self.get_parameter('language').value
        self.record_seconds = self.get_parameter('record_seconds').value
        self.silence_threshold = self.get_parameter('silence_threshold').value
        self.min_audio_level = self.get_parameter('min_audio_level').value
        self.consecutive_silence = self.get_parameter('consecutive_silence').value
        
        # 音频配置
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        
        self.get_logger().info("初始化语音识别节点...")
        
        # 加载Whisper模型
        try:
            import whisper
            self.get_logger().info(f"正在加载Whisper模型: {model_size}")
            self.model = whisper.load_model(model_size)
            self.get_logger().info("Whisper模型加载成功")
        except Exception as e:
            self.get_logger().error(f"模型加载失败: {e}")
            return
        
        # 初始化音频设备
        try:
            self.audio = pyaudio.PyAudio()
            
            # 查找可用的输入设备
            self.input_device_index = None
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    self.get_logger().info(f"找到音频输入设备: {info['name']}")
                    self.input_device_index = i
                    break
            
            if self.input_device_index is None:
                self.get_logger().error("未找到可用的音频输入设备")
                return
            
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.CHUNK
            )
            self.get_logger().info("音频设备初始化成功")
            
        except Exception as e:
            self.get_logger().error(f"音频设备初始化失败: {e}")
            return
        
        # 校准背景噪声
        self.background_noise_level = self.calibrate_noise()
        self.get_logger().info(f"背景噪声级别: {self.background_noise_level}")
        
        # 启动语音识别线程
        self.listening = True
        self.recording_thread = threading.Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.get_logger().info("语音识别节点已启动！")
        self.get_logger().info("系统会过滤背景噪声，只在检测到人声时进行识别")
    
    def calibrate_noise(self):
        """校准背景噪声水平"""
        self.get_logger().info("正在校准背景噪声...请保持安静")
        time.sleep(1)
        
        noise_samples = []
        for i in range(20):  # 采集20个样本
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data**2))
                noise_samples.append(rms)
            except:
                pass
        
        if noise_samples:
            avg_noise = np.mean(noise_samples)
            self.get_logger().info(f"背景噪声校准完成: {avg_noise:.2f}")
            return avg_noise
        else:
            self.get_logger().warning("背景噪声校准失败，使用默认值")
            return 300  # 默认值
    
    def get_audio_level(self, audio_data):
        """计算音频数据的音量级别"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return np.sqrt(np.mean(audio_array**2))
    
    def is_silence(self, audio_data, threshold_ratio=2.0):
        """检测是否为静音"""
        current_level = self.get_audio_level(audio_data)
        threshold = self.background_noise_level * threshold_ratio
        return current_level < threshold
    
    def recording_loop(self):
        """改进的录音和识别循环"""
        silence_counter = 0
        audio_buffer = []
        
        self.get_logger().info("等待语音输入...")
        
        while self.listening:
            try:
                # 读取音频数据
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                current_level = self.get_audio_level(data)
                
                # 检测是否有语音活动
                if not self.is_silence(data):
                    silence_counter = 0
                    audio_buffer.append(data)
                    self.get_logger().info(f"检测到语音活动，级别: {current_level:.2f}", throttle_duration_sec=5)
                else:
                    silence_counter += 1
                    if audio_buffer:  # 有缓冲数据但当前是静音
                        if silence_counter >= self.consecutive_silence:
                            # 检测到语音结束，进行处理
                            if len(audio_buffer) > 10:  # 确保有足够的音频数据
                                self.get_logger().info("检测到语音结束，开始识别...")
                                audio_data = b''.join(audio_buffer)
                                text = self.recognize_speech(audio_data)
                                
                                if text and text.strip():
                                    self.publish_text(text)
                                else:
                                    self.get_logger().info("未识别到有效内容")
                            
                            audio_buffer = []
                            silence_counter = 0
                            self.get_logger().info("等待下一次语音输入...")
                
                # 防止缓冲区过大
                if len(audio_buffer) > int(self.RATE / self.CHUNK * 10):  # 最多10秒
                    audio_buffer = audio_buffer[-int(self.RATE / self.CHUNK * 5):]  # 保留最后5秒
                
            except Exception as e:
                self.get_logger().error(f"录音错误: {e}")
                time.sleep(0.1)
    
    def recognize_speech(self, audio_data):
        """使用Whisper识别语音"""
        try:
            # 检查音频级别是否足够
            audio_level = self.get_audio_level(audio_data)
            if audio_level < self.min_audio_level:
                self.get_logger().info(f"音频级别过低 ({audio_level:.2f})，跳过识别")
                return None
            
            # 创建临时WAV文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = f.name
            
            # 保存音频数据
            self.save_wav(audio_data, temp_file)
            
            # 使用Whisper识别
            result = self.model.transcribe(
                temp_file,
                language=self.language,
                fp16=False
            )
            
            text = result['text'].strip()
            
            # 清理临时文件
            try:
                os.unlink(temp_file)
            except:
                pass
            
            return text
            
        except Exception as e:
            self.get_logger().error(f"语音识别错误: {e}")
            return None
    
    def save_wav(self, audio_data, filename):
        """保存为WAV文件"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(audio_data)
    
    def publish_text(self, text):
        """发布识别结果"""
        # 过滤掉明显是噪声的识别结果
        if len(text) > 50:  # 如果识别结果过长，可能是噪声
            self.get_logger().info("过滤可能的噪声识别结果")
            return
            
        msg = AudioText()
        msg.speaker_text = text
        self.text_publisher.publish(msg)
        self.get_logger().info(f"识别结果: {text}")
    
    def destroy_node(self):
        """清理资源"""
        self.get_logger().info("正在关闭节点...")
        self.listening = False
        
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        if hasattr(self, 'stream'):
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        
        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except:
                pass
        
        self.get_logger().info("节点已关闭")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ImprovedWhisperNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"节点运行错误: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()