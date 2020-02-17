from setuptools import setup, find_packages


setup(
    name="FNN",
    version="1.21",
    description='Simple Feedforward neural network for with last softmax layer for classification',
    author='kiru883',
    packages=find_packages(include=['utils.*', 'utils', 'utils.functions', 'utils.functions.*']),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ]
)
