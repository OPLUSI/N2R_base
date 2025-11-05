#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from n2r_topic_interfaces.msg import AudioText
import threading
import time
import os

class TestWhisperNode(Node):
    def __init__(self):
        super().__init__('test_whisper_node')
        
        # 创建发布者
        self.text_publisher = self.create_publisher(AudioText, 'audio_text', 10)
        
        self.get_logger().info("测试模式语音识别节点启动")
        
        # 加载Whisper模型
        try:
            import whisper
            self.model = whisper.load_model("base")
            self.get_logger().info("Whisper模型加载成功")
        except Exception as e:
            self.get_logger().error(f"模型加载失败: {e}")
            return
        
        # 启动测试循环
        self.testing = True
        self.test_thread = threading.Thread(target=self.test_loop)
        self.test_thread.start()
    
    def test_loop(self):
        """测试循环"""
        test_phrases = [
            "请帮我把中间的红色方块拿过来",
        ]
        
        for i, phrase in enumerate(test_phrases):
            if not self.testing:
                break
                
            self.get_logger().info(f"测试 {i+1}/{len(test_phrases)}: 模拟识别 '{phrase}'")
            
            # 发布模拟识别结果
            self.publish_text(phrase)
            
            time.sleep(5)  # 每5秒发布一次
        
        self.get_logger().info("所有测试完成")
    
    def publish_text(self, text):
        """发布识别结果"""
        msg = AudioText()
        msg.speaker_text = text
        self.text_publisher.publish(msg)
        self.get_logger().info(f"发布结果: {text}")
    
    def destroy_node(self):
        """清理资源"""
        self.testing = False
        if hasattr(self, 'test_thread') and self.test_thread.is_alive():
            self.test_thread.join(timeout=2.0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TestWhisperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()