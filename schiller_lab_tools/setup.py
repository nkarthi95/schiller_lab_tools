from setuptools import setup, find_packages

setup(
    name='schiller_lab_tools',  # Replace with your package name
    version='0.1.0',  # Replace with your version
    author='Nikhil Karthikeyan, Ulf Schiller',
    author_email='nkarthi@udel.edu',
    packages=["schiller_lab_tools"],
    description='Analysis tools utilized by the schiller lab to calculate properties of soft matter systems',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your_username/your_repository',  # Replace with your repo URL
    packages=find_packages(),  # Automatically find packages in your folder
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