from setuptools import setup

setup(name='cse547',
      version='0.0.3',
      description='Code to solve exercises for UW\'s CSE 547',
      url='https://github.com/ppham27/cse547',
      author='Philip Pham',
      author_email='philip.pham2@gmail.com',
      license='MIT',
      packages=['cse547'],
      package_data = {
          'cse547': ['LICENSE', 'README.md'],
      },
      zip_safe=False)
