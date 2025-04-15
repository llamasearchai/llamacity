from setuptools import find_packages, setup

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read long description
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="multi_modal_transformers",
    version="0.1.0",
    author="Nik Jois" "Nik Jois" "Nik Jois" "Nik Jois" "Nik Jois",
    author_email="nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai",
    description="Multi-Modal Transformer Architecture with Custom Attention Mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multi-modal-transformers",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.10.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "nbsphinx>=0.9.0",
        ],
    },
)
