"""
Written by Antoine Pare in April 2017
This code takes a 3DVSP and sorts it in function of headers
"""
import sys
import timeit
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

from obspy import Trace, Stream, UTCDateTime
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader, _read_segy

""" function defined below for the GUI """

def browse():
    global inputfile
    inputfile = askopenfilename()


def browseforsave():
    global outputfile
    outputfile = asksaveasfilename()


def select():
    if var_bool.get() == 'Ascending':
        ascending_order.append(True)
    else:
        ascending_order.append(False)
    
    headers_choosen.append(var_headers.get())



""" functions below to sort the SEG-Y file """

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

def sortSEGY():
    """ read input file """
    
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
    out.write(outputfile, format='SEGY', data_encoding=1, byteorder='big', textual_header_encoding='EBCDIC')
    
    root.destroy()




""" variables for the GUI """

headers_list = np.array(['trace_sequence_number_within_line',
       'trace_sequence_number_within_segy_file',
       'original_field_record_number',
       'trace_number_within_the_original_field_record',
       'energy_source_point_number', 'ensemble_number',
       'trace_number_within_the_ensemble', 'trace_identification_code',
       'number_of_vertically_summed_traces_yielding_this_trace',
       'number_of_horizontally_stacked_traces_yielding_this_trace',
       'data_use',
       'distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group',
       'receiver_group_elevation', 'surface_elevation_at_source',
       'source_depth_below_surface', 'datum_elevation_at_receiver_group',
       'datum_elevation_at_source', 'water_depth_at_source',
       'water_depth_at_group',
       'scalar_to_be_applied_to_all_elevations_and_depths',
       'scalar_to_be_applied_to_all_coordinates', 'source_coordinate_x',
       'source_coordinate_y', 'group_coordinate_x', 'group_coordinate_y',
       'coordinate_units', 'weathering_velocity', 'subweathering_velocity',
       'uphole_time_at_source_in_ms', 'uphole_time_at_group_in_ms',
       'source_static_correction_in_ms', 'group_static_correction_in_ms',
       'total_static_applied_in_ms', 'lag_time_A', 'lag_time_B',
       'delay_recording_time', 'mute_time_start_time_in_ms',
       'mute_time_end_time_in_ms', 'number_of_samples_in_this_trace',
       'sample_interval_in_ms_for_this_trace',
       'gain_type_of_field_instruments', 'instrument_gain_constant',
       'instrument_early_or_initial_gain', 'correlated',
       'sweep_frequency_at_start', 'sweep_frequency_at_end',
       'sweep_length_in_ms', 'sweep_type',
       'sweep_trace_taper_length_at_start_in_ms',
       'sweep_trace_taper_length_at_end_in_ms',
       'taper_type', 'alias_filter_frequency', 'alias_filter_slope',
       'notch_filter_frequency', 'notch_filter_slope', 'low_cut_frequency',
       'high_cut_frequency', 'low_cut_slope', 'high_cut_slope',
       'year_data_recorded', 'day_of_year', 'hour_of_day', 'minute_of_hour', 'second_of_minute',
       'time_basis_code', 'trace_weighting_factor',
       'geophone_group_number_of_roll_switch_position_one',
       'geophone_group_number_of_trace_number_one',
       'geophone_group_number_of_last_trace', 'gap_size',
       'over_travel_associated_with_taper',
       'x_coordinate_of_ensemble_position_of_this_trace',
       'y_coordinate_of_ensemble_position_of_this_trace',
       'for_3d_poststack_data_this_field_is_for_in_line_number',
       'for_3d_poststack_data_this_field_is_for_cross_line_number',
       'shotpoint_number', 'scalar_to_be_applied_to_the_shotpoint_number',
       'trace_value_measurement_unit', 'transduction_constant_mantissa',
       'transduction_constant_exponent', 'transduction_units', 'device_trace_identifier',
       'scalar_to_be_applied_to_times', 'source_type_orientation',
       'source_energy_direction_mantissa',
       'source_energy_direction_exponent', 'source_measurement_mantissa',
       'source_measurement_exponent', 'source_measurement_unit'])


global headers_choosen
global ascending_order

headers_choosen = []
ascending_order = []

""" GUI code """

root = tk.Tk()

var_headers = tk.StringVar(root)
var_headers.set('trace_sequence_number_within_line') # initial value
choices_headers = headers_list

var_bool = tk.StringVar(root)
var_bool.set('Ascending') # initial value
choices_bool = ['Ascending', 'Descending']


root.title("SEG-Y sort function")

root.geometry("1400x200+300+300")


browseButton = tk.Button(root, text='Select input SEG-Y', command=browse)
browseButton.pack(side=tk.TOP, padx=5, pady=20)

saveButton = tk.Button(root, text='Select output SEG-Y', command=browseforsave)
saveButton.pack(side=tk.TOP, padx=5, pady=5)

option_headers = tk.OptionMenu(root, var_headers, *choices_headers)
option_headers.pack(side=tk.LEFT, padx=10)

option_ascending = tk.OptionMenu(root, var_bool, *choices_bool)
option_ascending.pack(side=tk.LEFT, padx=10)

button_pickheader = tk.Button(root, text="Select this header to sort on", command=select)
button_pickheader.pack(side=tk.LEFT, padx=100)

button_sort = tk.Button(root, text="Sort the SEG-Y file", command=sortSEGY)
button_sort.pack(side=tk.LEFT, padx=20)

#text = tk.Text(root)
#text.pack()
#
#""" class to print in the text widget of the GUI instead of in the console """
#class StdoutRedirector(object):
#    def __init__(self,text_widget):
#        self.text_space = text_widget
#
#    def write(self,string):
#        self.text_space.insert('end', string)
#        self.text_space.see('end')
#
#sys.stdout = StdoutRedirector(text)

root.mainloop()

