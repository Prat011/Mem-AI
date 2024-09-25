from setuptools import setup, find_packages

setup(
    name="Mem-AI",
    version="0.1.0",
    author="Prathit Joshi",
    author_email="ppjoshi2100@gmail.com",
    description="A memory management library for Large Language Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Prathit-tech/Mem-AI",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "faiss-cpu",  # or faiss-gpu if GPU support is needed
    ],
    extras_require={
        "dev": ["pytest", "black", "isort"],
    },
)