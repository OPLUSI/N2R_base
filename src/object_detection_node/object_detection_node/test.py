#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import random

class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(3.0, self.timer_callback)  # 每3秒
        self.bridge = CvBridge()
        self.scene_counter = 0
        
        self.get_logger().info('测试图像发布器已启动，每3秒切换场景')
    
    def create_scene_1(self):
        """场景1：彩色圆形阵列"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (50, 50, 50)  # 深灰色背景
        
        # 创建彩色圆形网格
        colors = [
            ('red', (0, 0, 255)),
            ('green', (0, 255, 0)), 
            ('blue', (255, 0, 0)),
            ('yellow', (0, 255, 255)),
            ('purple', (255, 0, 255)),
            ('cyan', (255, 255, 0))
        ]
        
        rows, cols = 3, 4
        for i in range(rows):
            for j in range(cols):
                center_x = 80 + j * 150
                center_y = 80 + i * 120
                color_name, color = colors[(i * cols + j) % len(colors)]
                radius = 30 + random.randint(-10, 10)
                cv2.circle(image, (center_x, center_y), radius, color, -1)
        
        return image, "彩色圆形阵列"
    
    def create_scene_2(self):
        """场景2：彩色矩形阵列"""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 100  # 灰色背景
        
        colors = [
            ('dark_red', (0, 0, 200)),
            ('dark_green', (0, 200, 0)),
            ('dark_blue', (200, 0, 0)),
            ('orange', (0, 165, 255)),
            ('pink', (203, 192, 255)),
            ('brown', (42, 42, 165))
        ]
        
        # 创建矩形网格
        for i in range(4):
            for j in range(5):
                x = 50 + j * 120
                y = 50 + i * 100
                width = 60 + random.randint(-20, 20)
                height = 50 + random.randint(-15, 15)
                color_name, color = colors[(i * 5 + j) % len(colors)]
                cv2.rectangle(image, (x, y), (x + width, y + height), color, -1)
                # 添加边框
                cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 2)
        
        return image, "彩色矩形阵列"
    
    def create_scene_3(self):
        """场景3：混合形状"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (30, 30, 30)  # 黑色背景
        
        # 左侧：圆形
        circle_centers = [(100, 120), (100, 240), (100, 360)]
        circle_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for center, color in zip(circle_centers, circle_colors):
            cv2.circle(image, center, 40, color, -1)
            cv2.circle(image, center, 40, (255, 255, 255), 2)
        
        # 中间：矩形
        rects = [(250, 100, 80, 60), (250, 200, 70, 80), (250, 320, 90, 50)]
        rect_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for (x, y, w, h), color in zip(rects, rect_colors):
            cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # 右侧：三角形
        triangles = [
            [(450, 100), (500, 180), (400, 180)],
            [(450, 250), (500, 320), (400, 320)],
            [(450, 380), (500, 450), (400, 450)]
        ]
        triangle_colors = [(128, 0, 128), (0, 128, 128), (128, 128, 0)]
        for points, color in zip(triangles, triangle_colors):
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(image, [pts], color)
            cv2.polylines(image, [pts], True, (255, 255, 255), 2)
        
        return image, "混合形状（圆形+矩形+三角形）"
    
    def create_scene_4(self):
        """场景4：随机分布"""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 150  # 浅灰色背景
        
        # 随机生成10个形状
        for i in range(10):
            shape_type = random.choice(['circle', 'rectangle'])
            color = (
                random.randint(0, 255),
                random.randint(0, 255), 
                random.randint(0, 255)
            )
            
            if shape_type == 'circle':
                center = (random.randint(50, 590), random.randint(50, 430))
                radius = random.randint(20, 50)
                cv2.circle(image, center, radius, color, -1)
                cv2.circle(image, center, radius, (255, 255, 255), 2)
            else:  # rectangle
                x = random.randint(30, 570)
                y = random.randint(30, 420)
                w = random.randint(30, 100)
                h = random.randint(30, 80)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        return image, "随机分布形状"
    
    def create_scene_5(self):
        """场景5：同心圆和嵌套矩形"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (80, 80, 80)  # 灰色背景
        
        # 同心圆
        center = (320, 240)
        radii = [100, 70, 40]
        circle_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        
        for radius, color in zip(radii, circle_colors):
            cv2.circle(image, center, radius, color, 3)  # 只画轮廓
        
        # 嵌套矩形
        rect_center = (320, 240)
        sizes = [180, 120, 60]
        rect_colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for size, color in zip(sizes, rect_colors):
            x = rect_center[0] - size // 2
            y = rect_center[1] - size // 2
            cv2.rectangle(image, (x, y), (x + size, y + size), color, 3)
        
        return image, "同心圆和嵌套矩形"
    
    def timer_callback(self):
        """定时器回调，切换不同场景"""
        scenes = [
            self.create_scene_1,
            self.create_scene_2, 
            self.create_scene_3,
            self.create_scene_4,
            self.create_scene_5
        ]
        
        scene_func = scenes[self.scene_counter % len(scenes)]
        image, scene_name = scene_func()
        
        # 添加场景名称文本
        cv2.putText(image, f"Scene {self.scene_counter + 1}: {scene_name}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 发布图像
        msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        self.publisher_.publish(msg)
        
        self.get_logger().info(f'发布场景 {self.scene_counter + 1}: {scene_name}')
        self.scene_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()