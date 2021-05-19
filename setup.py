from setuptools import setup, find_packages



setup(
    name="archeml",
    author="mrp3anut",
   
    
    url="https://github.com/mrp3anut/archeml",
    
    packages=find_packages(),
    install_requires=[
    'numpy',
    'obspy'
    'scipy'
    'scikit-learn',
    'eqtransformer == 0.1.55'
    ], 
   )

