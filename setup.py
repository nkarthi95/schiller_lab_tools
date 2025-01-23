from setuptools import setup

setup(
    name='schiller_lab_tools',  # Replace with your package name
    version='0.1.0',  # Replace with your version
    author='Nikhil Karthikeyan, Ulf Schiller',
    author_email='nkarthi@udel.edu',
    description='Analysis tools utilized by the Schiller lab to calculate properties of soft matter systems',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nkarthi95/schiller_lab_tools',  # Replace with your repo URL
    packages=["schiller_lab_tools"],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Adjust based on your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Specify your Python version requirement
    install_requires=[
        "numpy <= 2.0",
    ]
)