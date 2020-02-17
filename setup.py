from setuptools import setup, find_packages


setup(
    name="FNN",
    version="1.0",
    description='Simple Feedforward neural network for with last softmax layer for classification',
    author='kiru883',
    packages=['FNN', 'FNN.functions'],
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ]
)
