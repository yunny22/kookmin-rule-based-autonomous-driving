from glob import glob
from setuptools import setup

package_name = "lane_seg_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/models", glob("models/README.md")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="TeamKAI public-release staging",
    maintainer_email="maintainer@example.invalid",
    description=(
        "LR-ASPP MobileNetV3 lane segmentation and canonical road perception"
    ),
    license="UNLICENSED",
    entry_points={
        "console_scripts": [
            (
                "lane_seg_lraspp_inference_node = "
                "lane_seg_control.lraspp_inference_node:main"
            ),
            "lane_seg_canonical_adapter = lane_seg_control.canonical_adapter_node:main",
        ],
    },
)
