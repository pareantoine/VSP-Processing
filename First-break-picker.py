# -*- coding: utf-8 -*-
"""
Written by Antoine Paré in April 2017

This code takes a 3DVSP stacked dataset and automatically pick the first break time to the onset
Save the input SEG-Y file to an output SEG-Y file with the first break in the headers
Save the first break time as an ascii file

usage : input_file output_file maximum_possible_FB_time

"""

import sys
import timeit

from scipy.interpolate import interp1d
import numpy as np

from segpy.dataset import DelegatingDataset
from segpy.reader import create_reader
from segpy.writer import write_segy

from segpy_numpy.extract import extract_trace_headers
from segpy.trace_header import TraceHeaderRev1



class DimensionalityError(Exception):
    pass


class TimedDataset(DelegatingDataset):

    def __init__(self, source_dataset, first_arrival_time):
        super().__init__(source_dataset)
        self._first_arrival_time = first_arrival_time
    
    #save first break time in the output file header
    def trace_header(self, trace_index):
        header = self.source.trace_header(trace_index)
        header.water_depth_at_group = self._first_arrival_time[trace_index]*100
        return header

#running window average
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

#function to measure first arrival time
def FBTime(segy_reader, maximum_fb_time):
    #extract necessary header
    sp, comp, vsp_md = extract_trace_headers(
            reader=segy_reader,
            fields=(TraceHeaderRev1.ensemble_trace_num,
                    TraceHeaderRev1.energy_source_point_num,
                    TraceHeaderRev1.datum_elevation_at_receiver_group))
    
    #calculate a few attributes of the VSP dataset
    nbr_of_geophone = np.count_nonzero(np.unique(vsp_md))
    nbr_of_SP = np.count_nonzero(np.unique(sp))
    nbr_of_components = np.count_nonzero(np.unique(comp))
    nbr_of_traces = nbr_of_geophone * nbr_of_SP * nbr_of_components
    sample_rate = int(segy_reader.trace_header(0).sample_interval/1000)
    
    #Check if attributes are giving the same number of traces as in the input file
    if nbr_of_traces == segy_reader.num_traces():
        print('All Shot Points, Components and Geophones accounted for.')
    else:
        nbr_of_traces = segy_reader.num_traces()
        print('Some traces are missing, please stop program and check SEG-Y file header')

    tic = timeit.default_timer() #to measure the time it takes to calculate the first break picking
                            
    #set up a series of variable for the automatic first break picker                          
    first_arrival_time = np.empty(0, dtype=np.int32) #empty array to store the first arrival time
    fore_window = 10 #10 samples in the fore window
    back_window = int(fore_window * 1.5) #15 samples in the back window
    
    #the first ratio measurement is done at the end of the back window, therefore we need to add 'back_window - 1' sample to the ratio array to get the correct time of the maximum ratio sample
    auto_pick_ratio_start = np.zeros(back_window - 1) 
    
    #set up a time array equal to 10 samples 
    time = np.linspace(0, 9, 10) - 5
    
    #Start measuring first arrival time, source-receiver pair by source-receiver pair    
    for i in range(int(nbr_of_traces/nbr_of_components)):
        #create a total vector trace by summing the absolute values of all three components for all samples
        data_to_vectorise = np.sum([np.absolute(segy_reader.trace_samples(i*nbr_of_components + j)) for j in range(nbr_of_components)], axis=0)
        data_to_vectorise = data_to_vectorise[0:int(maximum_fb_time/sample_rate)]
        #measure the average value in each window on the total vector trace
        mean_back_window = running_mean(data_to_vectorise, back_window)[:-fore_window]
        mean_fore_window = running_mean(data_to_vectorise, fore_window)[back_window:]
        
        #calculate the ratio between fore and back window
        auto_pick_ratio = mean_fore_window / mean_back_window
        
        #add to the ratio array the 'back_window' samples with couldn't not be measured to have an array with an index equal to the time sample in the data
        auto_pick_ratio = np.append(auto_pick_ratio_start, auto_pick_ratio)
        
        #extract the first arrival time precise to the nearest sample
        rough_first_arrival_time = auto_pick_ratio.argmax()
        
        # interpolate around the rough first arrival time and find a better first break time
        time_window = time + rough_first_arrival_time #time samples around the first arrival time -5/+4 samples
        start = int(time_window[0]) #first time sample
        end = int(time_window[-1]) #last time sample
        amplitude_window = data_to_vectorise[start:end+1] #extract the amplitude of the total vector trace at these time samples
        
        #set up the interpolation function to measure amplitude at any time with a cubic spline function
        f = interp1d(time_window, amplitude_window, kind='cubic')
        
        #interpolate every 0.1 sample
        interpolated_amp = np.array([f(t) for t in np.arange(start, end, 0.1)])
        
        #measure the average value in each window on the total vector trace
        mean_back_window = running_mean(interpolated_amp, back_window)[:-fore_window]
        mean_fore_window = running_mean(interpolated_amp, fore_window)[back_window:]
        
        #calculate the ratio between fore and back window
        auto_pick_ratio = mean_fore_window / mean_back_window
        
        #add the calculated first arrival time to the firt_arrival_time array n time (n = number of components)
        first_arrival_time = np.append(first_arrival_time, np.repeat(auto_pick_ratio.argmax()*sample_rate/10 + start*sample_rate, nbr_of_components))
        
        #every 2000 loops tell the users how much has been done of the calculation
        if i%2000==0:
            print('The automatic first break picking has done {0:.1f}% of the work, almost there!'.format(i/(nbr_of_SP*nbr_of_geophone)*100))

    tac = timeit.default_timer()
    print('The automatic first break picking took {0:.1f} seconds'.format(tac-tic))

    return first_arrival_time



def transform(in_filename, out_filename, maximum_fb_time):
    with open(in_filename, 'rb') as in_file, \
         open(out_filename, 'wb') as out_file:
        
        print('Loading SEG-Y file')
        segy_reader = create_reader(in_file)
        print('Calculation starts...')
        first_arrival_time = FBTime(segy_reader, maximum_fb_time)
        np.savetxt(in_filename[:-4]+'_FirstBreakTime.txt', first_arrival_time, fmt='%.2f')
        
        transformed_dataset = TimedDataset(segy_reader, first_arrival_time)
        print('Saving SEG-Y file')
        write_segy(out_file, transformed_dataset)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    in_filename = argv[0]
    out_filename = argv[1]
    maximum_fb_time = int(argv[2])
    
    transform(in_filename, out_filename, maximum_fb_time)

if __name__ == '__main__':
    sys.exit(main())
