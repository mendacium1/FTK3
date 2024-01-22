from setuptools import setup, find_packages

setup(
    name='si',
    version='1.1',
    packages=find_packages(),
    description='Package by Jürgen Fuß vor FTK3',
    long_description='Useful functions for DML1, GDK2, and FTK3',
    author='Jürgen Fuß',
    author_email='',
    url='https://github.com/mendacium1/FTK3',
    install_requires=[
        'numpy',
        'sympy',
        'alive-progress',
    ],
)

