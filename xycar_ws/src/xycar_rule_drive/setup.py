from glob import glob
import os

from setuptools import find_packages, setup


package_name = "xycar_rule_drive"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TeamKAI public-release staging",
    maintainer_email="maintainer@example.invalid",
    description="Public staging of the canonical Stanley-Pure Pursuit rule driver.",
    license="UNLICENSED",
    entry_points={
        "console_scripts": [
            "canonical_stanley_pursuit_driver = xycar_rule_drive.canonical_stanley_pursuit_driver:main",
        ],
    },
)
