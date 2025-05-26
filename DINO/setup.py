from setuptools import setup, find_packages

setup(
    name='dino_package',
    version='0.1',
    packages=find_packages(where='.'),  # This finds dino_package and its submodules
    install_requires=[
        'torch',
        'numpy',
        'Pillow',
    ]
)