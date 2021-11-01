# Bifrost Tensor Core Correlator Wrapper

### [![Paper](https://img.shields.io/badge/A%26A-Romein%202021-blue.svg)](https://doi.org/10.1051/0004-6361/202141896)

This provides a wrapper for Bifrost that allows the Tensor Core Correlator to be
used within a Bifrost pipeline.

To build the wrapper:
```
cd tensor-core-correlator
make
cd ../bifrost
python make_bifrost_plugin.py [--bifrost-path=...]
```

To use the wrapper:
```
$ python
>>> from bifrost import ndarray
>>> from btcc import Btcc
>>>
>>> bits_per_sample = 8
>>> ntime_per_gulp = 32
>>> nchan = 128
>>> nstand = 256
>>> npol = 2
>>>
>>> tcc = Btcc()
>>> tcc.init(bits_per_sample,
...          ntime_per_gulp,
...          nchan,
...          nstand,
...          npol)
>>>
>>> input_data = ndarray(shape=(ntime_per_gulp, nchan, nstand*npol),
...                      dtype='ci8',
...                      space='cuda')
>>> output_data = ndarray(shape=(nchan, nstand*(nstand+1)//2*npol*npol),
...                       dtype='ci32',
...                       space='cuda')
>>> dump = True
>>> tcc.execute(input_data, output_data, dump)
```

_NOTE:_ You may need to set LD_LIRBRARY_PATH or copy the `libtcc.so.5` file to your working directory.
