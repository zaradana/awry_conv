from setuptools import find_namespace_packages, setup

package_versions = [
    l.strip()
    for l in open("requirements.txt", "r").readlines()
    if not l.strip().startswith("#")
]

setup(
    name="awry_conv",
    version="2024.08.26",
    install_requires=package_versions,
    include_package_data=True,
    description="Predicting whether a conversation goes awry",
    options={
        "build_ext": {
            "define": "apex",
        }
    },
    long_description=open("README.md").read(),
)
