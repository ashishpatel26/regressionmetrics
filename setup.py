from setuptools import setup, find_packages
import os

with open("README.md", encoding="utf8") as f:
    long_description = f.read()

# Setting up
setup(
    name="regressionmetrics", 
    version="1.4.0",
    author="ashishpatel26",
    author_email="ashishpatel.ce.2011@gmail.com",
    description="Regression Metrics Calculation Made easy.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scikit-learn', 'tensorflow-cpu', 'tensorflow-gpu'],
    keywords=['regressionmertics','regression', 'metrics', 'regression metrics', 'regression metrics calculation', 'regression metrics calculation made easy'],
    url='http://github.com/ashishpatel26/regressionmetrics',
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    platforms=["any"],
    zip_safe=True,
)