from setuptools import setup
import os
from glob import glob

package_name = 'angle_processor'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        # Include video files in the package
        (os.path.join('share', package_name, 'data'), glob('angle_processor/*.mp4')),
        # Install video files to the package directory
        (os.path.join('lib', 'python3.10', 'site-packages', package_name), glob('angle_processor/*.mp4')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='phuc',
    maintainer_email='phuc@todo.todo',
    description='Angle Processor',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'angle_processor = angle_processor.angle_processor_node:main',
            'rotate_base_server = angle_processor.rotate_base_server:main',
        ],
    },
)
