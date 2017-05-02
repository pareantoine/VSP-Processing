"""
Written by Antoine Pare in April 2017
This code takes a 3DVSP and sorts it in function of headers
"""
import sys
import timeit
import numpy as np
import pandas as pd
import tkinter as tk

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

    
""" read input file """

inputfile = 'test.sgy'

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
    sys.exit('Some traces are missing, program stopped')


""" Choose the headers to sort on with a pop-up window which shows a drop down menu of all headers"""

headers_list = np.array([word.split(':')[0] for word in str(stream.traces[0].header).split('\n')])[:-1]
headers_choosen = []
ascending_order = []

def select():
    if var_bool.get() == 'Ascending':
        ascending_order.append(True)
    else:
        ascending_order.append(False)
    
    headers_choosen.append(var.get())

def finish():
    root.destroy()

root = tk.Tk()

# use width x height + x_offset + y_offset (no spaces!)
root.geometry("%dx%d+%d+%d" % (1000, 80, 100, 100))

root.title("tk.Optionmenu as combobox")

var = tk.StringVar(root)
# initial value
var.set('trace_sequence_number_within_line')
choices = headers_list

var_bool = tk.StringVar(root)
# initial value
var_bool.set('Ascending')
choices_bool = ['Ascending', 'Descending']

option = tk.OptionMenu(root, var, *choices)
option.pack(side='left', padx=10, pady=10)

option = tk.OptionMenu(root, var_bool, *choices_bool)
option.pack(side='left', padx=10, pady=10)

button = tk.Button(root, text="Select this header to sort on", command=select)
button.pack(side='left', padx=10, pady=10)

button = tk.Button(root, text="Sort", command=finish)
button.pack(side='left', padx=10, pady=10)
    
root.mainloop()


""" read in the headers and store them in a dataframe: df """

df = pd.DataFrame()

for headers in headers_choosen:   
    df[headers] = [getattr(t.header, headers) for t in stream.traces]

""" sort the dataframe accordingly """

df_sorted = df.sort_values(by=headers_choosen, ascending=ascending_order)


""" create a sorted out stream """
out = Stream()


print('Sorting the input file in progress')
      
""" save in the out stream the traces of input file in the sorted order """
tic = timeit.default_timer()

nbr_of_traces_to_sort = df_sorted.index.size
i = 0

for index in df_sorted.index:
    i += 1
    savetracetostream(stream, index, out, stream.traces[index].data)
    
    if i%2000==0:
        print('Sorting the input file in progress: {0:.1f}%'.format(i/nbr_of_traces_to_sort*100))

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
