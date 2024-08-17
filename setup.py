from setuptools import setup, find_packages


setup(
    name='torch_1k', 
    version='0.0.2', 
    packages=find_packages(),
    description='Mini-PyTorch within 1000 lines of code',
    install_requires = ['numpy', 'loguru'],
    scripts=[],
    python_requires = '>=3',
    include_package_data=True,
    author='Liu Shengli',
    url='http://github.com/gseismic/torch_1k',
    zip_safe=False,
    author_email='liushengli203@163.com'
)
