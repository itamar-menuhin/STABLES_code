from setuptools import setup, find_packages

setup(
    name="model_improvement",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here
        "numpy",
        "pandas",
        "scikit-learn",
        # "tensorflow", or "pytorch", etc.
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A project for improving machine learning models",
    keywords="machine learning, model improvement",
    python_requires=">=3.6",
)
