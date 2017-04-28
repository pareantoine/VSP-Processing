# -*- coding: utf-8 -*-
"""
Written by Antoine Paré in April 2017

This code takes a 3DVSP stacked dataset and automatically pick the first break time

"""
import timeit
import sys
import numpy as np
from scipy.interpolate import interp1d


from obspy.io.segy.segy import _read_segy


def FBTime(inputfile):
    
    # Read the input SEG-Y files, 3D-VSP stacked file, sorted in SP, MD, Trace Number (output of stack)
    # Save it at a stream object, the seismic format from the Obspy library
    
    tic = timeit.default_timer()
    print('The SEG-Y files is loading ... Please be patient')
    stream = _read_segy(inputfile, headonly=True)
    toc = timeit.default_timer()
    print('The loading took {0:.0f} seconds'.format(toc-tic))
    
    # Create a series of variable to be able to save each component of each geophone
    # We will automatically figure out the following, number of geophone, number of shot points, number of components
    
    nbr_of_geophone = np.count_nonzero(np.unique(np.array([t.header.datum_elevation_at_receiver_group for t in stream.traces])))
       
    nbr_of_SP = np.count_nonzero(np.unique(np.array([t.header.energy_source_point_number for t in stream.traces])))
    
    nbr_of_components = np.count_nonzero(np.unique(np.array([t.header.trace_number_within_the_ensemble for t in stream.traces])))
        
    # We QC the variables by checking that the calculated number of traces is equal to the number of traces in the stream object (input SEG-Y file)
    
    nbr_of_traces = nbr_of_geophone * nbr_of_SP * nbr_of_components
    
    #nbr_of_samples_per_trace = stream.binary_file_header.number_of_samples_per_data_trace
    
    if nbr_of_traces == int(str(stream).split()[0]):
        print('All Shot Points, Components and Geophones accounted for.')
    else:
        print('Some traces are missing, please stop program and check SEG-Y file header')
    
    
    # running_mean is a function with calculates the average in a moving window
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 
    
    
    tic = timeit.default_timer() #to measure the time it takes to calculate the first break picking
    
                            
    #set up a series of variable for the automatic first break picker                          
    first_arrival_time = np.empty(0, dtype=np.int32) #empty array to store the first arrival time
    fore_window = 10 #10 samples in the fore window
    back_window = int(fore_window * 1.5) #15 samples in the back window
    
    #the first ratio measurement is done at the end of the back window, therefore we need to add 'back_window - 1' sample to the ratio array to get the correct time of the maximum ratio sample
    auto_pick_ratio_start = np.zeros(back_window - 1) 
    
    #set up a time array equal to 4 samples 
    time = np.linspace(0, 3, 4) - 2
    
    #Start measuring first arrival time, source-receiver pair by source-receiver pair    
    for i in range(nbr_of_SP*nbr_of_geophone):
        #create a total vector trace by summing the absolute values of all three components for all samples
        data_to_vectorise = np.sum([np.absolute(stream.traces[i*nbr_of_components + j].data) for j in range(nbr_of_components)], axis=0)

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
        
        #interpolate every 0.1ms
        interpolated_amp = np.array([f(t) for t in np.arange(start, end, 0.1)])
        
        #measure the average value in each window on the total vector trace
        mean_back_window = running_mean(interpolated_amp, back_window)[:-fore_window]
        mean_fore_window = running_mean(interpolated_amp, fore_window)[back_window:]
        
        #calculate the ratio between fore and back window
        auto_pick_ratio = mean_fore_window / mean_back_window
        ##### previous interpolation was running too slowly with 100 samples, went down to 40 samples, which now takes about 3ms (+5ms to read the data in the first place)
        
        #add the calculated first arrival time to the firt_arrival_time array n time (n = number of components)
        first_arrival_time = np.append(first_arrival_time, np.repeat(auto_pick_ratio.argmax()/10 + start, nbr_of_components))
        
        #every 2000 loops tell the users how much has been done of the calculation
        if i%2000==0:
            print('The automatic first break picking has done {0:.1f}% of the work, almost there!'.format(i/(nbr_of_SP*nbr_of_geophone)*100))
    
    tac = timeit.default_timer()
    print('The automatic first break picking took {0:.1f} seconds'.format(tac-tic))
    
    # Save the first arrival time to a text file to import it later on
    np.savetxt('FirstBreakTime.txt', first_arrival_time, fmt='%.2f')



    #Write first arrival time to headers of input SEG-Y file

#    for t in range(nbr_of_traces):
#        stream.traces[t].header.water_depth_at_group = int(first_arrival_time[t]*100)
#   
#                                                   
#    #output SEG-Y with the first break time in headers
#    print('Saving SEG-Y files...')
#    stream.write('FirstBreakPicked.segy', data_encoding=1, endian='>')
 

def main(infile):
    FBTime(infile)                                                    


if __name__ == "__main__":
    main(sys.argv[1])