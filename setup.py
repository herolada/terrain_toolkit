import os
from glob import glob

from setuptools import setup

package_name = "terrain_toolkit_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[
        "terrain_toolkit",
        "terrain_toolkit.heightmap",
        "terrain_toolkit.traversability",
        "terrain_toolkit.outlier",
        "terrain_toolkit.icp",
        "terrain_toolkit_ros",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ales Kucera",
    maintainer_email="kuceral4@fel.cvut.cz",
    description="ROS 2 wrapper for the terrain_toolkit GPU terrain pipeline.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "terrain_toolkit_node = terrain_toolkit_ros.terrain_toolkit_node:main",
        ],
    },
)
