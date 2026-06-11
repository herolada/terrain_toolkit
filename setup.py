from setuptools import setup, find_packages

setup(
    name="terrain-toolkit",
    version="0.1.0",
    description="Add your description here",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.12",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.21.0",
        "warp-lang>=1.12.1",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.10.8",
            "plotly>=6.7.0",
        ],
    },
)