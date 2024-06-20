## Requirements & Compilation

1. Requirements

Required packages are listed in [requirements.txt](requirements.txt). 

The code is tested using Python-3.8.5 with PyTorch 1.7.1.

2. Compile extra modules

```shell script
cd network/knn_search
python setup.py build_ext --inplace
cd ../pointnet2_ext
python setup.py build_ext --inplace
cd ../../utils/extend_utils
python build_extend_utils_cffi.py
```
According to your installation path of CUDA, you may need to revise the variables cuda_version in [build_extend_utils_cffi.py](utils/extend_utils/build_extend_utils_cffi.py).

