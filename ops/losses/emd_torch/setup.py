from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
	name='PyTorch EMD',
	version='0.0',
	author='Vinit Sarode',
	author_email='vinitsarode5@gmail.com', 
	description='A PyTorch module for the earth mover\'s distance loss', 
	ext_package='_emd_ext',
	ext_modules=[
		CUDAExtension(
			name='_emd', 
			sources=[
				'pkg/src/emd.cpp',
				'pkg/src/cuda/emd.cu',
			],
			include_dirs=['pkg/include'],
		),
	],
	packages=[
		'emd', 
	],
	package_dir={
		'emd' : 'pkg/layer'
	},
	cmdclass={'build_ext': BuildExtension}, 
)