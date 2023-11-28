from setuptools import setup, find_packages

from canary_sefi.copyright import get_system_version


def get_install_requires():
    reqs = [
        'torch >= 2.0.1',
        'torchvision >= 0.16.1',
        'colorama >= 0.4.6',
        'tqdm >= 4.66.0',
        'flask >= 3.0.0',
        'flask-socketio >= 5.3.6',
        'matplotlib >= 3.7.4',
        'opencv-python >= 4.8.1.78',
        'piq == 0.7.0',
        'pandas >= 2.0.3',
        'seaborn >= 0.13.0',
        'scikit-learn >= 1.3.2',
    ]
    return reqs


setup(
    name='canary_sefi',  # 项目名称
    version=get_system_version(),  # 版本
    author='jiazheng.sun',  # 作者
    author_email='jiazheng.sun@foxmail.com',  # 联系邮箱
    url='https://github.com/NeoSunJZ/Canary_Master',  # 项目url
    description='Canary SEFI is a framework for evaluating the adversarial robustness of deep learning-based image recognition models.',
    # 简单描述
    long_description=open('readme.md', encoding="utf-8").read(),
    long_description_content_type="text/markdown",  # 长描述格式
    packages=find_packages(exclude=["canary_lib.*", "canary_lib"]),  # 要打包的package
    # packages=find_packages(),
    # package_data={'pystopwords': ['data/*.txt']},  # 要包含的数据文件
    python_requires=">=3.8",
    install_requires=get_install_requires(),  # 本库依赖的库
    license='Apache',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
