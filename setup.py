from setuptools import setup, find_packages
import os

here = os.path.dirname(os.path.abspath(__file__))


def requirements():
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
        return f.read().splitlines()


def version(*parts):
    version_file = os.path.join(*parts)
    version_ns = {}

    with open(version_file) as f:
        exec(f.read(), {}, version_ns)

    return version_ns['__version__']

setup(
    name='chunky3d',
    description='A 3D array-like NumPy-based data structure for large sparsely-populated volumes',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    version=version(here, 'chunky3d', '_version.py'),
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
        # currently missing deps
        # 'Programming Language :: Python :: 3.8',
    ]
    , project_urls={
        'Source': 'https://github.com/K3D-tools/chunky3d',
        'Tracker': 'https://github.com/K3D-tools/chunky3d/issues',
    }
)
