import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import requests
from threading import Lock

class WebCamSubscriber(Node):
    def __init__(self):
        super().__init__('webcam_subscriber')
        
        # 创建发布者
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        
        # Windows Flask服务器地址
        self.stream_url = "http://192.168.85.144:5000/video_feed"
        
        # 初始化视频捕获
        self.cap = None
        self.frame_lock = Lock()
        self.current_frame = None
        
        self.get_logger().info(f'Connecting to webcam stream: {self.stream_url}')
        
        # 尝试连接视频流
        self.connect_to_stream()
        
        # 创建定时器发布图像
        self.timer = self.create_timer(0.033, self.timer_callback)  # ~30Hz
        
    def connect_to_stream(self):
        """连接到视频流"""
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap.isOpened():
                self.get_logger().info('Successfully connected to webcam stream')
            else:
                self.get_logger().error('Failed to open stream')
        except Exception as e:
            self.get_logger().error(f'Error connecting to stream: {str(e)}')
    
    def timer_callback(self):
        """定时器回调函数，发布图像"""
        if self.cap is None or not self.cap.isOpened():
            self.connect_to_stream()
            return
            
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # 转换为ROS2图像消息
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                ros_image.header.stamp = self.get_clock().now().to_msg()
                ros_image.header.frame_id = "webcam_frame"
                
                # 发布图像
                self.publisher_.publish(ros_image)
                self.get_logger().info('Publishing webcam frame', throttle_duration_sec=2.0)
            else:
                self.get_logger().warning('Failed to read frame from stream')
                # 尝试重新连接
                self.cap.release()
                self.connect_to_stream()
                
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {str(e)}')
            # 发生错误时尝试重新连接
            if self.cap:
                self.cap.release()
                self.connect_to_stream()

def main(args=None):
    rclpy.init(args=args)
    node = WebCamSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 清理资源
        if node.cap:
            node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()