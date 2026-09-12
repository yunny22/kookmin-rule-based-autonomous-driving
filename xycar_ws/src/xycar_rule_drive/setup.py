from setuptools import find_packages, setup


package_name = "xycar_rule_drive"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TeamKAI",
    maintainer_email="team.kai@example.com",
    description="Sanitized Pure Pursuit–Stanley RULE controller for a portfolio release.",
    license="UNLICENSED",
    entry_points={
        "console_scripts": [
            "canonical_stanley_pursuit_driver = xycar_rule_drive.canonical_stanley_pursuit_driver:main",
        ],
    },
)
