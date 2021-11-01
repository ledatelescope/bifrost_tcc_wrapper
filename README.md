# Bifrost Tensor Core Correlator Wrapper

### [!Paper](https://img.shields.io/badge/A%26Ap-Romein%202021-blue.svg)](https://www.aanda.org/articles/aa/pdf/forth/aa41896-21.pdf)

This provides a wrapper for Bifrost that allows the Tensor Core Correlator to be
used within a Bifrost pipeline.

To use:
```
cd tensor-core-correlator
make
cd ../bifrost
python make_bifrost_plugin.py [--bifrost-path=...]
```
