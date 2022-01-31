import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorflow_quant",
    version="0.0.1",
    author="howardya",
    author_email="howardya@github.com",
    description="Quant Libraries Using Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/howardya/tensorflow_quant",
    project_urls={
        "Bug Tracker": "https://github.com/howardya/tensorflow_quant/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "tensorflow_quant"},
    packages=setuptools.find_packages(where="tensorflow_quant"),
    python_requires=">=3.6",
)
