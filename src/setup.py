import setuptools
import os

loc = os.path.abspath(os.path.dirname(__file__))

setuptools.setup(
        name = "behavioral_autoencoder",
        version = "0.0.1",
        author = "Balint Kurgyis and Taiga Abe",
        author_email = "",
        description = "standalone autoencoder code for behavioral video",
        long_description = "",
        long_description_content_type = "test/markdown", 
        url = "https://github.com/cellistigs/alm_2pimagingAnalysis",
        packages = setuptools.find_packages(),
        include_package_data=True,
        package_data={},
        classifiers = [
            "License :: OSI Approved :: MIT License"],
        python_requires=">=3.7",
        )



