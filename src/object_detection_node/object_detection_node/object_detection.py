import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from n2r_topic_interfaces.msg import CvResult
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        self.bridge = CvBridge()
        
        # 订阅摄像头图像
        self.image_sub = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.image_callback, 
            10
        )
        
        # 发布带有标记的图像和坐标字符串
        self.marked_image_pub = self.create_publisher(CvResult, '/marked_image', 10)
        
        # 参数声明
        self.declare_parameter('min_area', 500)  # 最小轮廓面积
        self.declare_parameter('max_area', 50000)  # 最大轮廓面积
        
        self.get_logger().info('轮廓检测物体识别节点已启动')
    
    def detect_objects_by_contour(self, cv_image):
        """通过轮廓检测物体"""
        detections = []
        
        # 转换为灰度图
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用自适应阈值二值化
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 形态学操作，连接断开的轮廓
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取参数
        min_area = self.get_parameter('min_area').get_parameter_value().integer_value
        max_area = self.get_parameter('max_area').get_parameter_value().integer_value
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 过滤面积过大或过小的轮廓
            if min_area <= area <= max_area:
                # 计算轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算轮廓的中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    center_x = x + w // 2
                    center_y = y + h // 2
                
                detections.append({
                    'contour': contour,
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': area
                })
        
        # 按面积从大到小排序
        detections.sort(key=lambda x: x['area'], reverse=True)
        
        return detections
    
    def generate_coordinate_string(self, detections):
        """生成坐标列表字符串"""
        if not detections:
            return "未检测到物体"
        
        coord_strings = []
        for i, detection in enumerate(detections):
            center_x, center_y = detection['center']
            coord_strings.append(f"物体{i+1}:({center_x},{center_y})")
        
        return " | ".join(coord_strings)
    
    def draw_markers(self, cv_image, detections):
        """在图像上只标记物品编号"""
        marked_image = cv_image.copy()
        height, width = marked_image.shape[:2]
        
        # 预定义颜色列表
        colors = [
            (0, 255, 255),   # 黄色
            (255, 0, 255),   # 粉色
            (255, 255, 0),   # 青色
            (0, 255, 0),     # 绿色
            (255, 0, 0),     # 蓝色
            (0, 0, 255),     # 红色
            (255, 165, 0),   # 橙色
            (128, 0, 128),   # 紫色
            (0, 128, 128),   # 深青色
            (255, 192, 203), # 浅粉色
            (173, 216, 230), # 浅蓝色
            (144, 238, 144), # 浅绿色
        ]
        
        for i, detection in enumerate(detections):
            center_x, center_y = detection['center']
            x, y, w, h = detection['bbox']
            contour = detection['contour']
            
            # 为物体分配ID和颜色
            object_id = i + 1
            color = colors[i % len(colors)]
            
            # 绘制轮廓
            cv2.drawContours(marked_image, [contour], -1, color, 2)
            
            # 绘制中心点十字标记
            cross_size = max(min(w, h) // 4, 8)
            cv2.line(marked_image, 
                    (center_x - cross_size, center_y), 
                    (center_x + cross_size, center_y), 
                    color, 2)
            cv2.line(marked_image, 
                    (center_x, center_y - cross_size), 
                    (center_x, center_y + cross_size), 
                    color, 2)
            
            # 绘制中心点圆点
            cv2.circle(marked_image, (center_x, center_y), 6, color, -1)
            
            # 只标记物品编号（简化显示）
            label = f"{object_id}"
            font_scale = 0.8
            thickness = 2
            
            # 计算文字大小
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # 在物体中心上方显示编号
            text_x = center_x - text_size[0] // 2
            text_y = center_y - 15
            
            # 确保文本在图像范围内
            text_x = max(10, min(text_x, width - text_size[0] - 10))
            text_y = max(text_size[1] + 10, min(text_y, height - 10))
            
            # 绘制文字背景
            cv2.rectangle(marked_image, 
                         (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5), 
                         (0, 0, 0), -1)
            
            # 绘制编号文字
            cv2.putText(marked_image, label, 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return marked_image
    
    def image_callback(self, msg):
        """处理接收到的图像"""
        try:
            # 转换ROS图像为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 通过轮廓检测物体
            detections = self.detect_objects_by_contour(cv_image)
            
            # 生成坐标字符串
            coord_string = self.generate_coordinate_string(detections)
            
            # 在图像上绘制标记（只显示编号）
            marked_image = self.draw_markers(cv_image, detections)
            
            # 创建CvResult消息
            cv_result_msg = CvResult()
            
            # 设置图像字段
            cv_result_msg.res = self.bridge.cv2_to_imgmsg(marked_image, 'bgr8')
            cv_result_msg.res.header = msg.header
            
            # 设置坐标字符串字段
            cv_result_msg.res_xy_list = coord_string
            
            # 发布CvResult消息
            self.marked_image_pub.publish(cv_result_msg)
            
            # 记录检测信息
            if detections:
                self.get_logger().info(f'检测到 {len(detections)} 个物体')
                self.get_logger().info(f'坐标字符串: {coord_string}')
            else:
                self.get_logger().info('未检测到物体')
                
        except Exception as e:
            self.get_logger().error(f'图像处理错误: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()