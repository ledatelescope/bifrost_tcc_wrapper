
# Copyright (c) 2023, The Bifrost Authors. All rights reserved.
# Copyright (c) 2023, The University of New Mexico. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#   
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import unittest
import numpy as np
import os
import sys
import shutil

from bifrost import ndarray, zeros as bfzeros
from bifrost.quantize import quantize
from bifrost.unpack import unpack

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btcc import Btcc


class tcc_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the btcc module."""
    
    @staticmethod
    def compute_gold(idata, npol=2, decim=1):
        try:
            _, nchan, nstand, npol = idata.shape
        except ValueError:
            _, nchan, nstandpol = idata.shape
            nstand = nstandpol // npol
            
        if decim > 1:
            idata = idata.reshape(-1, nchan//decim, decim, nstand*npol)
            idata = idata.sum(axis=2)
            nchan //= decim
            
        odata = np.zeros((nchan,nstand*(nstand+1)//2*npol*npol), dtype=np.complex64)
        for c in range(nchan):
            k = 0
            for i in range(nstand):
                tic = idata[:,c,:].real - 1j*idata[:,c,:].imag  # Manually conjugate the first signal
                for j in range(i, nstand):
                    for p0 in range(npol):
                        for p1 in range(npol):
                            odata[c,npol*npol*k+npol*p0+p1] += (tic[:,npol*i+p0]*idata[:,c,npol*j+p1]).sum()
                    k += 1
                    
        return odata
        
    
    @staticmethod
    def create_data(nbit=8, ntime=64, nchan=32, nstand=16, npol=2):
        data = np.random.rand(ntime,nchan,nstand,npol) \
               + 1j*np.random.rand(ntime,nchan,nstand,npol)
        data -= 0.5 + 0.5j
        data *= 2**(nbit-1)-1
        data = data.astype(np.complex64)
        
        if nbit == 4:
            dtype = 'ci4'
        elif nbit == 8:
            dtype = 'ci8'
        else:
            dtype = 'cf16'
            
        if nbit == 8:
            data /= 16
        
        data = ndarray(data, space='system')
        qdata = ndarray(shape=data.shape, dtype=dtype, native=False, space='system')
        quantize(data, qdata, scale=1.0)
        return qdata.copy(space='cuda')
        
    def run_test(self, nbit=8, ntime=64, nchan=32, nstand=16, npol=2, decim=1, naccum=1):
        idata = self.create_data(nbit=nbit, ntime=ntime, nchan=nchan, nstand=nstand, npol=npol)
        cc = Btcc()
        cc.init(nbit, ntime, nchan, nstand, npol, decim)
        
        idata = idata.reshape(ntime,nchan,nstand*npol)
        odata = bfzeros(shape=(nchan//decim,nstand*(nstand+1)//2*npol*npol), dtype='ci32', space='cuda')
        for i in range(2):
            for j in range(naccum):
                cc.execute(idata, odata, j == (naccum-1))
        odata = odata.copy(space='system')
        odata = odata['re'] + 1j*odata['im']
        
        idata = idata.copy(space='system')
        try:
            idata = idata['re'] + 1j*idata['im']
        except ValueError:
            idata = np.int8(idata['re_im'] & 0xF0) + 1j*np.int8((idata['re_im'] & 0x0F) << 4)
            idata /= 16
        odata_gold = self.compute_gold(idata, npol=npol, decim=decim)
        np.testing.assert_equal(odata, odata_gold*naccum)
        
    def test_tcc_ci4(self):
        self.run_test(nbit=4)
        
    def test_tcc_ci4_decim(self):
        self.run_test(nbit=4, decim=4)
        
    def test_tcc_ci4_accum(self):
        self.run_test(nbit=4, naccum=3)
        
    def test_tcc_ci4_decim_accum(self):
        for nstand in (16, 32, 64):
            with self.subTest(nstand=nstand):
                self.run_test(nbit=4, nstand=nstand, decim=4, naccum=3)
                
    def test_tcc_ci8(self):
        self.run_test(nbit=8)
        
    def test_tcc_ci8_decim(self):
        self.run_test(nbit=8, decim=4)
        
    def test_tcc_ci8_accum(self):
        self.run_test(nbit=8, naccum=3)
        
    def test_tcc_ci8_decim_accum(self):
        for nstand in (16, 32, 64):
            with self.subTest(nstand=nstand):
                self.run_test(nbit=8, nstand=nstand, decim=4, naccum=3)
        


class tcc_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the btcc units 
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(tcc_tests)) 


if __name__ == '__main__':
    unittest.main()
