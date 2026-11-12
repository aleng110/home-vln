from setuptools import setup
from glob import glob

setup(
    name='home_explore',
    version='0.1.0',
    packages=['home_explore'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/home_explore']),
        ('share/home_explore', ['package.xml']),
        ('share/home_explore/launch', glob('launch/*.py')),
        ('share/home_explore/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@example.com',
    description='Online household exploration and language-driven object inspection',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'agent=home_explore.agent:main',
            'guard=home_explore.guard:main',
            'cli=home_explore.cli:main',
            'viewer=home_explore.viewer:main',
            'scan_adapter=home_explore.scan_adapter:main',
            'map_adapter=home_explore.map_adapter:main',
            'obstacle_cloud=home_explore.obstacle_cloud:main',
        ]
    },
)
