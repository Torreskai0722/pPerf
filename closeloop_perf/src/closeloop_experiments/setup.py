from glob import glob
import os

from setuptools import find_packages, setup


PACKAGE_NAME = "closeloop_experiments"


setup(
    name=PACKAGE_NAME,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + PACKAGE_NAME]),
        ("share/" + PACKAGE_NAME, ["package.xml"]),
        (os.path.join("share", PACKAGE_NAME, "schema"),
         glob("closeloop_experiments/schema/*.json")),
    ],
    package_data={"closeloop_experiments": ["schema/*.json"]},
    install_requires=["setuptools", "PyYAML", "jsonschema"],
    zip_safe=True,
    maintainer="pPerf maintainers",
    maintainer_email="maintainers@example.com",
    description="Closed-loop run and campaign orchestration.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "validate = closeloop_experiments.cli:validate_main",
            "run = closeloop_experiments.cli:run_main",
            "campaign = closeloop_experiments.cli:campaign_main",
            "migrate = closeloop_experiments.cli:migrate_main",
        ],
    },
)
