from glob import glob
import os
from setuptools import find_packages, setup


package_name = "xycar_map_nav"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TeamKAI",
    maintainer_email="team.kai@example.com",
    description="Sanitized final mission arbitration and command-gate source.",
    license="UNLICENSED",
    entry_points={
        "console_scripts": [
            "sequential_hybrid_driver = xycar_map_nav.sequential_hybrid_driver:main",
            "space_drive_gate = xycar_map_nav.space_drive_gate:main",
        ],
    },
)
