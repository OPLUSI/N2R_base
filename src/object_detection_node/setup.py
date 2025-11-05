from setuptools import find_packages, setup

package_name = 'object_detection_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oplusi',
    maintainer_email='oplusi@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node = object_detection_node.object_detection:main',
            'test_object_detection_node = object_detection_node.test:main',
            'convert_res_node = object_detection_node.convert:main',
            'computer_camera_test_node = object_detection_node.computer_camera_test:main'
        ],
    },
)
