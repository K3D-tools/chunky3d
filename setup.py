from setuptools import setup, find_packages
import os


def requirements():

    with open(os.path.join(os.path.dirname(__file__),'requirements.txt')) as f:
        return f.read().splitlines()

setup(
    name="chunky3d",
    version='0.0.1',
    license='MIT',
    install_requires=requirements(),
    packages=find_packages(),
    keywords=['3d', 'array', 'chunked', 'sparse'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Multimedia :: Graphics :: 3D Modeling',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)
