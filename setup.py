from setuptools import setup, find_packages
from typing import List

def get_requirement(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and req.strip() != '-e .']
    return requirements

setup(
    name="car_sales_project",
    version="0.0.1",
    author="Dawit Tsegaye",
    author_email="dawit.tsegaye-ug@aau.edu.et",
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)
