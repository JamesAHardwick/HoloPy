from setuptools import setup, find_packages

# pip install . (install package locally) 

# pip install -e . (install the package with a symlink – changes to the source files will be immediately available)

setup(
    name='algorithms',
    version='0.1.0',
    author='James Hardwick',
    author_email='',
    packages=find_packages(),
    # packages=find_packages(include=['algorithms', 'algorithms.*']),
    setup_requires=['pytest-runner'],
    tests_require=['pytest']                    
)