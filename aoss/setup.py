import os
import shutil
import subprocess

import setuptools

# try:
#     git_describe = subprocess.check_output(
#         ['git', 'describe', '--tags', '--long']).decode('utf-8').strip()
#     git_branch = subprocess.check_output(
#         ['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8').strip()
#     version = f'{git_describe}-{git_branch}'
#
#     with open('aoss_client/version.py', 'w') as f:
#         f.write(f"version = '{version}'\n")
#         f.truncate()
# except Exception:
#     from importlib.machinery import SourceFileLoader
#     version_module = SourceFileLoader(
#         'version_module', 'aoss_client/version.py').load_module()
#     version = version_module.version
#
# dist_path = 'dist'
# if os.path.exists(dist_path):
#     shutil.rmtree(dist_path)

# version指定tag比如: v2.2.5
version = "v2.2.6"

setuptools.setup(
    name="aoss-python-sdk",
    version=version,
    description="Aoss S3 storage API for Pytorch, Parrots",
    url="https://gitlab.bj.sensetime.com/elementary/quark/quarkoss-python-sdk",
    packages=setuptools.find_packages(),
    package_data={"": ["**/*.so"]},
    install_requires=["boto3", "coloredlogs", "environs", "humanize", "multiprocessing-logging"],
    python_requires=">=3.6",
    zip_safe=False,
)
