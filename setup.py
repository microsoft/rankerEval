import io
import os

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return fd.read()


setup(
    name="rankerEval",
    version="0.1.1",
    url="https://github.com/microsoft/rankerEval",
    license='MIT',

    author="Tobias Schnabel",
    author_email="tobias.schnabel@microsoft.com",

    description="A fast numpy-based implementation of ranking metrics for information retrieval and recommendation.",
    keywords='ranking recommendation evaluation map mrr',
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",
    packages=find_packages(exclude=('tests', 'docs', 'test')),

    install_requires=['numpy>=1.18'],
    python_requires='>=3',

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
