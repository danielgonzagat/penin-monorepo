#!/usr/bin/env python3
"""
Setup script for PENIN Evolution System
Sistema de IA com arquitetura modular e auto-evolução
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Read version from version file
def get_version():
    version_file = Path(__file__).parent / "penin" / "__version__.py"
    if version_file.exists():
        with open(version_file, 'r') as f:
            exec(f.read())
            return locals()['__version__']
    return "2.0.0"

# Read long description from README
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "PENIN Evolution System - Sistema de IA com arquitetura modular e auto-evolução"

# Read requirements from requirements.txt
def get_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    return []

# Development requirements
dev_requirements = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "black>=23.7.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.4.0",
    "pre-commit>=3.3.0",
    "bandit>=1.7.0",
    "jupyter>=1.0.0",
    "ipython>=8.14.0",
]

# Documentation requirements
doc_requirements = [
    "sphinx>=7.1.0",
    "sphinx-rtd-theme>=1.3.0",
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.1.0",
]

# ML/AI specific requirements
ml_requirements = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "tensorflow>=2.13.0",
    "scikit-learn>=1.3.0",
    "huggingface-hub>=0.15.0",
    "spacy>=3.6.0",
    "sentence-transformers>=2.2.0",
]

# API requirements
api_requirements = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.22.0",
    "pydantic>=2.0.0",
    "sqlalchemy>=2.0.0",
    "redis>=4.5.0",
]

setup(
    name="penin-evolution-system",
    version=get_version(),
    author="PENIN Team",
    author_email="dev@penin.ai",
    description="Sistema de IA com arquitetura modular e capacidades de auto-evolução",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/danielgonzagat/penin-monorepo",
    project_urls={
        "Bug Tracker": "https://github.com/danielgonzagat/penin-monorepo/issues",
        "Documentation": "https://penin-docs.readthedocs.io/",
        "Source Code": "https://github.com/danielgonzagat/penin-monorepo",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": dev_requirements,
        "docs": doc_requirements,
        "ml": ml_requirements,
        "api": api_requirements,
        "all": dev_requirements + doc_requirements + ml_requirements + api_requirements,
    },
    entry_points={
        "console_scripts": [
            "penin=penin.cli:main",
            "penin-server=penin.api.server:run",
            "penin-neural=penin.neural.cli:main",
            "penin-omega=penin.omega.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "penin": [
            "config/*.yaml",
            "config/*.json",
            "templates/*.html",
            "static/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "artificial-intelligence",
        "machine-learning",
        "neural-networks",
        "auto-evolution",
        "modular-architecture",
        "nlp",
        "deep-learning",
        "fastapi",
        "python",
    ],
    platforms=["any"],
    license="MIT",
    
    # Custom commands
    cmdclass={},
    
    # Additional metadata
    maintainer="PENIN Evolution Team",
    maintainer_email="evolution@penin.ai",
    
    # Test suite
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
    ],
)

# Post-install hooks
def post_install():
    """Execute post-installation tasks"""
    print("🚀 PENIN Evolution System installed successfully!")
    print("📖 Quick start guide:")
    print("   1. Initialize: penin init")
    print("   2. Configure: edit config/system_config.yaml")
    print("   3. Start server: penin-server")
    print("   4. Run neural core: penin-neural")
    print("🔗 Documentation: https://penin-docs.readthedocs.io/")

if __name__ == "__main__":
    setup()
    
    # Run post-install if this is an install command
    if "install" in sys.argv:
        post_install()