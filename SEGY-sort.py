"""
Written by Antoine ParĂŠ in April 2017

This code takes a 3DVSP after first break picking and perform a statistical horizontal rotation

"""
import sys
import timeit
import numpy as np
import math
import statsmodels.api as sm
import pandas as pd

from obspy import read, Trace, Stream, UTCDateTime
from obspy.core import AttribDict, Stats
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader, _read_segy

def savetracetostream(stream, idx, out, data):
    """
    this function add data (data) using the header from stream (idx) to an output stream object (out)
    delta needs to be changed to the sample rate of data (not automatically done yet)
    """
    np.require(data, dtype=np.float32)
    trace = Trace(data=data)
    trace.stats.delta = 0.01
    trace.stats.starttime = UTCDateTime(2011,11,11,11,11,11)
    if not hasattr(trace.stats, 'segy.trace_header'): 
                trace.stats.segy = {}        
    trace.stats.segy.trace_header = SEGYTraceHeader() 
    trace.stats.segy.trace_header = stream.traces[idx].header
    out.append(trace) 

def choose_header_to_sort():
    """
    this function displays a list of headers available for sorting and asks
    the user to choose which header to sort on and wether to perform an
    ascending or descending sort (should be a bolean)
    """
    
    headers_list = np.array([word.split(':')[0] for word in str(stream.traces[0].header).split('\n')])[:-1]
    
    headers_choosen = np.array([])
    ascending_order = np.array([])
    
    stop = 0
    i = 1 
    while stop < 1:
        """ ask user the number of the header he would like to choose """
        print('With which header would you like to do the sort number {}'.format(i))
        m = 0
        for headers in headers_list[0:10]:
            print(m + ': ' + headers + '/n')
            m+=1
        choice = input('With which header would you like to do the sort number {}, n for next 10, q to exit'.format(i))
        if choice == 'q':
            stop = 1
        elif choice == 'n':
         """ need to fill in the options with the next 10 headers or other alternative """   
        else:
            headers_choosen = np.append(headers_choosen, choice)
            
            order_pick = input('1: Ascending, 2:Descending')
            if order_pick == 1:
                order = True
            else:
                order = False
            ascending_order = np.append(ascending_order, order)
            
            i = i+1
    
    return headers_choosen, ascending_order
 
    

inputfile = 'KOC_FB.sgy'

print('The SEG-Y files is loading ... Please be patient')
tic = timeit.default_timer()
stream = _read_segy(inputfile, headonly=True)
toc = timeit.default_timer()
print('The loading took {0:.0f} seconds'.format(toc-tic))

""" create a list of all the geophones' depth """
geophone_depths = (np.array([t.header.datum_elevation_at_receiver_group for t in stream.traces])/100)[::3]                       

""" Calculate the number of tools, shot points and components in the 3D-VSP, then check that the calculated number of traces 
is equal to the number of traces in the SEG-Y file """  
nbr_of_geophone = np.count_nonzero(np.unique(geophone_depths))
nbr_of_SP = np.count_nonzero(np.unique(np.array([t.header.energy_source_point_number for t in stream.traces])))
nbr_of_components = np.count_nonzero(np.unique(np.array([t.header.trace_number_within_the_ensemble for t in stream.traces])))
nbr_of_traces = nbr_of_geophone * nbr_of_SP * nbr_of_components

if nbr_of_traces == int(str(stream).split()[0]):
    print('All Shot Points, Components and Geophones accounted for.')
else:
    print('Some traces are missing, please stop program and check SEG-Y file header')

        
    
"""
to sort a SEG-Y data, I will need to save all headers requeted for the sort in
a pandas dataframe with columns in the order of the header sort. The trace index being
the index of the dataframe which can be outputed at the end.
Then use pandas to sort the columns as requested, output the trace index column.
Create the out stream by reading the input file trace by trace as specified in the 
dataframe.

This create a list of all the headers:
headers_list = np.array([word.split(':')[0] for word in str(stream.traces[0].header).split('\n')])

example of sorting a dataframe: 
df_sorted = df.sort_values(by=['geophone_depths','SP','components'], ascending=[True,False,True])
getting the index:
df_sorted.index[i]

""" 



""" create a sorted out stream """
out = Stream()

tic = timeit.default_timer()

for index in df_sorted.index:
    savetracetostream(stream, index, out, stream.traces[index].data)
    
    if index%2000==0:
        print('Rotation of the input file in progress: {0:.1f}%'.format(i/df_sorted.index.size*100))

toc = timeit.default_timer()
print('The sort took {0:.0f} seconds'.format(toc-tic))


""" set up the necessary textual and binary file header for the output stream: out """
out.stats = AttribDict()
#out.stats = Stats(dict(textual_file_header=stream.textual_file_header))
out.stats.textual_file_header = stream.textual_file_header
out.stats.binary_file_header = stream.binary_file_header
out.stats.binary_file_header.trace_sorting_code = stream.binary_file_header.trace_sorting_code


""" QC of input/output """
print('The input SEG-Y had {} traces'.format(str(stream).split()[0]))
print('After applying the rotation module...')
print('The output SEG-Y has {} traces'.format(out.count()))

""" save 'out' as a SEG-Y file """
print('Saving SEG-Y files...')
out.write('Sorted.sgy', format='SEGY', data_encoding=1, byteorder='big', textual_header_encoding='EBCDIC')
