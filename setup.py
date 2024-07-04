from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
    print(f"Final requirements list: {requirements}")
    
    return requirements


setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='NutriScan is a comprehensive data pipeline project that analyzes the intricate relationship between food accessibility, retail food environment, and health outcomes across U.S. counties. Leveraging the USDAs Food Environment Atlas dataset, this project aims to provide insights into how the food landscape correlates with critical health indicators such as diabetes and obesity rates.',
    author='Gaurav Kumar Chaurasiya',
    license='',
    install_requires=get_requirements('requirements.txt'),
)