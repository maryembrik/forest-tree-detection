from setuptools import setup, find_packages

setup(
    name="forest-tree-detection",
    version="1.0.0",
    description="Automatic Detection of Dead & Diseased Trees from Satellite Imagery",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.15",
        "albumentations>=1.3",
        "scikit-learn>=1.3",
        "opencv-python>=4.8",
        "numpy>=1.24",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "tqdm>=4.65",
        "streamlit>=1.28",
        "plotly>=5.17",
        "reportlab>=4.0",
        "Pillow>=10.0",
        "PyYAML>=6.0",
    ],
)
