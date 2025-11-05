import rclpy
from rclpy.node import Node
from n2r_topic_interfaces.msg import CvResult
from n2r_topic_interfaces.msg import AudioText
from n2r_topic_interfaces.msg import Command
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import base64
import json
import requests
import threading
from typing import Optional
import re

class VLMNode(Node):
    def __init__(self):
        super().__init__('vlm_node')
        
        # 参数配置 - 通义千问
        self.declare_parameter('api_key', 'sk-a52e781f6aa44dc6aa5d1968415f543b')
        self.declare_parameter('api_url', 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation')
        self.declare_parameter('model', 'qwen-vl-plus')  # qwen-vl-plus, qwen-vl-max
        self.declare_parameter('max_tokens', 1500)
        
        # CV bridge
        self.bridge = CvBridge()
        
        # 存储当前的图像和文本
        self.current_image = None
        self.current_text = ""
        self.current_list = ""
        self.lock = threading.Lock()
        
        # 订阅者 - cv输入
        self.cv_sub = self.create_subscription(
            CvResult,
            '/marked_image',
            self.cv_callback,
            10
        )
        
        # 订阅者 - 文本输入
        self.text_sub = self.create_subscription(
            AudioText,
            'audio_text',
            self.text_callback,
            10
        )
        
        # 订阅者 - 触发处理
        self.trigger_sub = self.create_subscription(
            String,
            '/vlm/trigger',
            self.trigger_callback,
            10
        )
        
        # 发布者 - VLM结果
        self.result_pub = self.create_publisher(
            Command,
            '/vlm/result',
            10
        )
        
        self.get_logger().info('VLM节点已启动，等待图像和文本输入...')
        
    def cv_callback(self, msg):
        """接收图像输入"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg.res, "bgr8")
            with self.lock:
                self.current_image = cv_image
                self.current_list  = msg.res_xy_list
            self.get_logger().info('收到cv输入')
            
        except Exception as e:
            self.get_logger().error(f'图像处理错误: {str(e)}')
    
    def text_callback(self, msg):
        """接收文本输入"""
        with self.lock:
            self.current_text = msg.speaker_text
        self.get_logger().info(f'收到文本输入: {msg.speaker_text}')

    def trigger_callback(self, msg):
        """触发VLM处理"""
        self.get_logger().info('收到处理触发信号')
        self.process_vlm()
    
    def process_vlm(self):
        """使用VLM处理图像和文本"""
        with self.lock:
            if self.current_image is None:
                self.get_logger().warn('没有可用的图像数据')
                return
            
            if not self.current_text:
                self.get_logger().warn('没有可用的文本数据')
                return
            
            image = self.current_image.copy()
            text = self.current_text
        
        # 在新线程中处理避免阻塞
        thread = threading.Thread(
            target=self._call_vlm_api,
            args=(image, text)
        )
        thread.start()
    
    def _call_vlm_api(self, image, text_prompt):
        """调用VLM API - 只输出坐标版本"""
        try:
            # 编码图像为base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.get_parameter('api_key').value}"
            }
            
            # 强制要求只输出坐标的提示词
            coordinate_prompt = text_prompt + self.current_list + "。请只输出坐标，格式为(x,y)，不要包含任何其他文字、说明或符号。"
            
            # 通义千问格式
            payload = {
                "model": self.get_parameter('model').value,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": f"data:image/jpeg;base64,{image_base64}"},
                                {"text": coordinate_prompt}
                            ]
                        }
                    ]
                },
                "parameters": {
                    "max_tokens": self.get_parameter('max_tokens').value
                }
            }
            
            self.get_logger().info(f'调用VLM API，提示词: {coordinate_prompt}')
            
            response = requests.post(
                self.get_parameter('api_url').value,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 通义千问的响应格式解析
                if 'output' in result and 'choices' in result['output']:
                    content = result['output']['choices'][0]['message']['content']
                    
                    # 提取坐标
                    coordinates = self._extract_coordinates(content)
                    
                    # 发布结果 - 坐标直接放入location字段
                    result_msg = Command()
                    result_msg.action = "navigate"
                    result_msg.object = "target"  # 可以固定或根据情况设置
                    result_msg.location = coordinates  # 坐标直接放入location
                    
                    self.result_pub.publish(result_msg)
                    self.get_logger().info('VLM处理完成')
                    self.get_logger().info(f'提取的坐标: {coordinates}')
                    
                else:
                    raise Exception(f'API响应格式错误: {result}')
                    
            else:
                error_msg = f'API请求失败: {response.status_code} - {response.text}'
                self.get_logger().error(error_msg)
                
                # 发布错误信息
                error_result = Command()
                error_result.action = "error"
                error_result.object = "API error"
                error_result.location = "0,0"  # 错误时的默认坐标
                self.result_pub.publish(error_result)
                
        except Exception as e:
            error_msg = f'VLM处理错误: {str(e)}'
            self.get_logger().error(error_msg)

            error_result = Command()
            error_result.action = "error"
            error_result.object = "processing error"
            error_result.location = "0,0"  # 错误时的默认坐标
            self.result_pub.publish(error_result)

    def _extract_coordinates(self, content):
        """从内容中提取坐标"""
        try:
            # 解析内容
            parsed_content = self._parse_qwen_content(content)
            
            # 使用正则表达式提取坐标
            
            # 匹配 (x,y) 或 x,y 格式的坐标
            patterns = [
                r'\((\d+),\s*(\d+)\)',  # (x,y)
                r'(\d+),\s*(\d+)',      # x,y
                r'坐标.*?(\d+).*?(\d+)', # 坐标 x y
                r'位置.*?(\d+).*?(\d+)'  # 位置 x y
            ]
            
            for pattern in patterns:
                matches = re.search(pattern, parsed_content)
                if matches:
                    x = matches.group(1)
                    y = matches.group(2)
                    return f"({x},{y})"
            
            # 如果没有找到坐标，返回默认值或原始内容的前20个字符
            if len(parsed_content) <= 20:
                return parsed_content
            else:
                return "0,0"  # 默认坐标
                
        except Exception as e:
            self.get_logger().error(f'坐标提取失败: {str(e)}')
            return "0,0"  # 错误时的默认坐标

    def _parse_qwen_content(self, content):
        """解析通义千问返回的内容"""
        try:
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                return ' '.join(text_parts)
            elif isinstance(content, dict):
                if 'text' in content:
                    return content['text']
                else:
                    return str(content)
            elif isinstance(content, str):
                return content
            else:
                return str(content)
        except Exception as e:
            return str(content)

def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()