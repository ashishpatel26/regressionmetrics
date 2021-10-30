from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

VERSION = '0.1.2'
DESCRIPTION = 'Regression Metrics Calculation Made easy'
LONG_DESCRIPTION = 'This package contains all regression metrics and work with scikit learn and tensorflow.'

# Setting up
setup(
    name="regmetrics",
    version=VERSION,
    author="ashishpatel26",
    author_email="ashishpatel.ce.2011@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scikit-learn', 'tensorflow-cpu==2.6.0', 'tensorflow-gpu==2.6.0'],
    url='http://github.com/ashishpatel26/regmetrics',
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    zip_safe=True,
)
