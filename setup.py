from setuptools import setup, find_packages

setup(
    name="vae",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "torch",
        "torchmetrics",
        "ipdb",
    ],
)
