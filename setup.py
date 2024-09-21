from setuptools import setup, find_packages

setup(
    name='flax-pilot',
    version='0.1.13',
    author='Nithish M',
    author_email='nithishm2206@gmail.com',
    description='A Simplistic trainer for Flax',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NITHISHM2410/flax-pilot',
    packages=find_packages(include=['fpilot', 'fpilot.*']),
    install_requires=[
        'jax[cuda12]>=0.4.26',
        'flax>=0.8.4',
        'optax',
        'numpy',
        'tqdm',
        'mergedeep',
        'typing',
        'orbax-checkpoint>=0.5.15',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
