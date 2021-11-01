# Bifrost Tensor Core Correlator Wrapper

### [![Paper](https://img.shields.io/badge/A%26A-Romein%202021-blue.svg)](https://doi.org/10.1051/0004-6361/202141896)

This provides a wrapper for Bifrost that allows the Tensor Core Correlator to be
used within a Bifrost pipeline.

To use:
```
cd tensor-core-correlator
make
cd ../bifrost
python make_bifrost_plugin.py [--bifrost-path=...]
```
