
from setuptools import setup, find_packages


setup(
    name='funcobspy',
    version='1.0.0',
    author='Hassan A. Kingravi',
    author_email='hkingravi@gmail.com',
    description='Function observers in Python',
    url='https://github.com/hkingravi/funcobspy',
    packages=find_packages(exclude=['*.test', 'test']),
    install_requires=[]
)
