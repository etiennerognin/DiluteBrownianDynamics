from setuptools import setup, find_packages

setup(
    name='dilutebrowniandynamics',
    version='1.0',
    description='Dilute Brownian Dynamics',
    author='Etienne Rognin',
    author_email='ecr43@cam.ac.uk',
    packages=find_packages(include=['dilutebrowniandynamics', 'dilutebrowniandynamics.*'])
)
