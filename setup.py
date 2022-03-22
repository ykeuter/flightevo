import setuptools

setuptools.setup(
    name="flightevo",
    version="0.0.1",
    author="Yvo Keuter",
    author_email="ypmkeuter@gmail.com",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)