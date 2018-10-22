from setuptools import find_packages, setup

setup(
    name='gym_trading',
    version='0.0.2',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['gym', 'matplotlib', 'TA-Lib'])
