from setuptools import find_packages, setup
setup(
    name="MoleCare-ML",
    version="0.0.1",
    author="Yauhen Bichel",
    author_email="info@molecare.co.uk",
    description="Machine Learning RESTful service for melanoma prediction",
    url="https://github.com/MoleCare/ml-rest-service",
    project_urls={

    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        'flask',
    ],
)