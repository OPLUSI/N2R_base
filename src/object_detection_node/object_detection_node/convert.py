import rclpy
from rclpy.node import Node
from n2r_topic_interfaces.msg import CvResult
from sensor_msgs.msg import Image

class ResultConverter(Node):
    def __init__(self):
        super().__init__('result_converter')
        
        # 订阅CvResult
        self.subscription = self.create_subscription(
            CvResult,
            '/marked_image',
            self.result_callback,
            10
        )
        
        # 发布纯Image
        self.image_pub = self.create_publisher(Image, '/marked_image_view', 10)
        
        self.get_logger().info('CvResult转换节点已启动')
    
    def result_callback(self, msg):
        # 直接从CvResult中提取Image消息并转发
        self.image_pub.publish(msg.res)
        self.get_logger().info('已转发标记图像')

def main(args=None):
    rclpy.init(args=args)
    node = ResultConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()