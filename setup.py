from setuptools import find_packages, setup

setup(
    name='face_tools',  # Важно: именно так, с подчеркиванием!
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    # Добавьте это для лучшей совместимости
    include_package_data=True,
)