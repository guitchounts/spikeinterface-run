from setuptools import setup, find_packages

setup(
    name='spikeinterface-run',
    packages=find_packages(),
    platforms=['mac', 'unix'],
    install_requires=['click', 'tqdm', 'numpy'],
    python_requires='>=3.8',
    entry_points={'console_scripts': ['spikeinterface-run = spikeinterface_run.main:cli']}
)