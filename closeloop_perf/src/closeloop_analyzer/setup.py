from setuptools import find_packages, setup


PACKAGE_NAME = 'closeloop_analyzer'


setup(
    name=PACKAGE_NAME,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + PACKAGE_NAME]),
        ('share/' + PACKAGE_NAME, ['package.xml']),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='pPerf maintainers',
    maintainer_email='maintainers@example.com',
    description='Offline analyzers for closed-loop profiler artifacts.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'analyze = closeloop_analyzer.cli:main',
        ],
    },
)
