from setuptools import setup, find_packages

setup(
    name='email-spam-detector',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'tkinter',  # GUI toolkit
    ],
    entry_points={
        'console_scripts': [
            'email-spam-detector=app:main',  # Assuming main function in app.py
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A simple email spam detection application.',
    keywords='spam detection email machine learning',
    url='https://github.com/yourusername/email-spam-detector',  # Replace with your repo URL
)