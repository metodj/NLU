from setuptools import setup, find_packages

setup(
        name='nlu-project2',
        version='0.1',
        description='Skeleton code for NLU Story Close Project',

        author='Rok Sikonja',
        author_email='rsikonja@student.ethz.ch',

        packages=find_packages(exclude=[]),
        python_requires='>=3.6',
        install_requires=[
                'tensorflow-gpu==2.9.3',
                'numpy',
                'pandas',
                'bert-tensorflow==1.0.1',
                'vaderSentiment',
                'tensorflow-hub',
                'nltk'
        ],
)