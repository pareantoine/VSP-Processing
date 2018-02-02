# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:56:38 2018

@author: Antoine Paré
"""

"""
Down sample VSP data in depth domain instead of time domain
""


import numpy as np
from scipy import signal

from segpy.dataset import DelegatingDataset
from segpy.reader import create_reader
from segpy.writer import write_segy

from segpy_numpy.extract import extract_trace_headers
from segpy.trace_header import TraceHeaderRev1

from itertools import islice

""" A simple progress indicator while running the code """
def make_progress_indicator(name):

    previous_integer_progress = -1

    def progress(p):
        nonlocal previous_integer_progress
        percent = p * 100.0
        current_integer_progress = int(percent)
        if current_integer_progress != previous_integer_progress:
            print("{} : {}%".format(name, current_integer_progress))
        previous_integer_progress = current_integer_progress

    return progress


class DimensionalityError(Exception):
    pass

""" Class to slice the input dataset to keep one trace ever n traces
n being the downsampling factor """
class SlicedDataset(DelegatingDataset):

    def __init__(self, source_dataset, downsampling_factor):
        super().__init__(source_dataset)
        self._downsampling_factor = downsampling_factor

    def trace_indexes(self):
        return islice(self._source.trace_indexes(), 0, None, self._downsampling_factor)
    

""" Class to take the sliced dataset and change samples to the downsampled samples and corresponding headers """
class DownSampledDataset(DelegatingDataset):

    def __init__(self, source_dataset, down_data, downsampling_factor):
        super().__init__(source_dataset)
        self._down_data = down_data
        self._downsampling_factor = downsampling_factor
    
    def trace_indexes(self):
        return iter(list(range(int(self.source.num_traces()/self._downsampling_factor))))
    
    def trace_samples(self, trace_index, start=None, stop=None):
        return self._down_data[trace_index]
    
    def trace_header(self, trace_index):
        return self._source.trace_header(trace_index*self._downsampling_factor)


""" Downsampling function """
def _downsample(segy_reader, factor):
    #extract the needed header values (every three traces)
    sp, vsp_md, = extract_trace_headers(
        reader=segy_reader,
        fields=(TraceHeaderRev1.energy_source_point_num,
                TraceHeaderRev1.das_measured_depth),
        trace_indexes=segy_reader.trace_indexes())
    
    #calculate the necessary attributes
    nbr_of_geophone = np.count_nonzero(np.unique(vsp_md))
    nbr_of_SP = np.count_nonzero(np.unique(sp))
    nbr_of_traces = nbr_of_geophone * nbr_of_SP

    #check if attributes match number of traces of the input file
    if nbr_of_traces == segy_reader.num_traces():
        print('All Shot Points and Geophones accounted for.')
    else:
        nbr_of_traces = segy_reader.num_traces()
        print('Some traces are missing, please stop program and check SEG-Y file header')
    
    #create empty array with correct number of samples, this will be used to store the downsampled data
    data_down = np.array([], dtype=np.float32).reshape(0,segy_reader.max_num_trace_samples())
    
    #downsample each shotpoint and store them in data_down
    for i in range(nbr_of_SP):
       data = np.stack(segy_reader.trace_samples(x) for x in range(i*nbr_of_geophone, (i+1)*nbr_of_geophone)) 
       data_down = np.concatenate((data_down, signal.decimate(data, factor, axis=0, zero_phase=True)), axis=0)
    
    return data_down


def transform(in_filename, out_filename, downsampling_factor):
    with open(in_filename, 'rb') as in_file, \
         open(out_filename, 'wb') as out_file:
        
        segy_reader = create_reader(in_file, progress=make_progress_indicator("Loading SEG-Y file"))
        print('Calculation starts...')
        down_data = _downsample(segy_reader, downsampling_factor)
        sliced_dataset = SlicedDataset(segy_reader, downsampling_factor)
        transformed_dataset = DownSampledDataset(sliced_dataset, down_data, downsampling_factor)
        write_segy(out_file, transformed_dataset, progress=make_progress_indicator("Saving SEG-Y file"))
