from setuptools import setup, find_packages


setup(
    name='flax-pilot',
    version='0.2.5',
    author='Nithish M',
    author_email='nithishm2206@gmail.com',
    description='A Simplistic trainer for Flax',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NITHISHM2410/flax-pilot',
    packages=find_packages(include=['fpilot', 'fpilot.*']),
    install_requires=[
        "jax[cuda12]==0.4.37",
        'flax>=0.10.2',
        'optax>=0.2.4',
        'numpy',
        'tqdm',
        'mergedeep',
        'orbax-checkpoint>=0.5.20',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.10',
)
