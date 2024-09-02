from setuptools import setup

setup(
	name='ipfs_accelerate_py',
	version='0.0.1',
	packages=[
        'ipfs_accelerate_py',
	],
	install_requires=[
        'ipfs_transformers_py',
        'ipfs_model_manager_py',
        'transformers',
		'torch',
        'torchvision',
        'numpy',
        'torchtext',
		'urllib3',
		'requests',
		'boto3',
        'toml',
        'websocket-client',
        'trio',
        'multiaddr',
	]
)