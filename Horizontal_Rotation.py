"""
Written by Antoine Paré in April 2017

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

#function that extract data around the FB of SEG-Y file (Stream) at trace idx, from window_min samples before FB to window_max samples after FB
def getfbdata(stream, idx, window_min, window_max):
    fb = int(stream.traces[idx].header.water_depth_at_group/100)
    return stream.traces[idx].data[(fb - int(window_min)) : (fb + (window_max) + 1)]

#thie function inputs the three component data around the FB and calculates the rotation angle to maximise the energy on Y with the same polarity of Z
def getangle(xcomp, ycomp, zcomp):
    model = sm.OLS(xcomp, ycomp)
    results = model.fit()
    angle = round((360 + math.degrees(math.atan2(results.params[0],1))) % 360, 2)
    ycomp_rot = xcomp * math.sin(math.radians(angle)) + ycomp * math.cos(math.radians(angle))
    if np.sign(ycomp_rot)[7] != np.sign(zcomp)[7]:
        if angle <= 180: 
            angle = angle + 180
        else:
            angle = angle - 180
    return angle

def savetracetostream(stream, idx, out, data):
    np.require(data, dtype=np.float32)
    trace = Trace(data=data)
    trace.stats.delta = 0.01
    trace.stats.starttime = UTCDateTime(2011,11,11,11,11,11)
    if not hasattr(trace.stats, 'segy.trace_header'): 
                trace.stats.segy = {}        
    trace.stats.segy.trace_header = SEGYTraceHeader() 
    trace.stats.segy.trace_header = stream.traces[idx].header 
    out.append(trace) 

    

inputfile = 'KOC_FB.sgy'

print('The SEG-Y files is loading ... Please be patient')
tic = timeit.default_timer()
stream = _read_segy(inputfile, headonly=True)
toc = timeit.default_timer()
print('The loading took {0:.0f} seconds'.format(toc-tic))


source_x = np.array([t.header.source_coordinate_x for t in stream.traces])[::3]
source_y = np.array([t.header.source_coordinate_y for t in stream.traces])[::3]
geo_x = np.array([t.header.group_coordinate_x for t in stream.traces])[::3]
geo_y = np.array([t.header.group_coordinate_x for t in stream.traces])[::3]

azimuth = np.round([(360 + math.degrees(math.atan2((source_x[t] - geo_x[t]), (source_y[t] - geo_y[t])))) % 360 for t in range(source_x.size)], decimals=2)

geophone_depths = (np.array([t.header.datum_elevation_at_receiver_group for t in stream.traces])/100)[::3]                       
                          
nbr_of_geophone = np.count_nonzero(np.unique(geophone_depths))
nbr_of_SP = np.count_nonzero(np.unique(np.array([t.header.energy_source_point_number for t in stream.traces])))
nbr_of_components = np.count_nonzero(np.unique(np.array([t.header.trace_number_within_the_ensemble for t in stream.traces])))
nbr_of_traces = nbr_of_geophone * nbr_of_SP * nbr_of_components

if nbr_of_traces == int(str(stream).split()[0]):
    print('All Shot Points, Components and Geophones accounted for.')
else:
    print('Some traces are missing, please stop program and check SEG-Y file header')

horizontal_angles = np.array([])
tool_face = np.array([])

for i,j in zip(range(nbr_of_SP*nbr_of_geophone), azimuth):
    xcomp = getfbdata(stream, i*3, 5, 10)
    ycomp = getfbdata(stream, i*3 +1, 5, 10)
    zcomp = getfbdata(stream, i*3 +2, 5, 10)
    horizontal_angles_i = getangle(xcomp, ycomp, zcomp)
    horizontal_angles = np.append(horizontal_angles, horizontal_angles_i)
    tool_face_i = j - horizontal_angles_i
    if tool_face_i < 0:
        tool_face_i = tool_face_i + 360
    elif tool_face_i > 360:
        tool_face_i = tool_face_i - 360
    tool_face = np.append(tool_face, tool_face_i)
    
    if i%2000==0:
        print('The horizontal rotation calculation is in progress: {0:.1f}%'.format(i/(nbr_of_SP*nbr_of_geophone)*100))

table_angle = np.column_stack((geophone_depths, azimuth, horizontal_angles, tool_face))
df = pd.DataFrame(table_angle, columns=['MD', 'shot azimuth', 'rotation angle', 'tool face'])

#need to add a better measurement of the angle to rotate, not using .mean() but using numpy.histogram
rotation_angle
angle_to_rotate = azimuth - np.tile(df.groupby('MD')['tool face'].values, nbr_of_SP)

out = Stream()

for i,j in zip(range(nbr_of_SP*nbr_of_geophone),angle_to_rotate):
    xcomp = stream.traces[i*3].data
    ycomp = stream.traces[i*3+1].data
    zcomp = stream.traces[i*3+2].data
    xcomp_rot = np.require(xcomp * math.cos(math.radians(j)) - ycomp * math.sin(math.radians(j)), dtype=np.float32)
    ycomp_rot = np.require(xcomp * math.sin(math.radians(j)) + ycomp * math.cos(math.radians(j)), dtype=np.float32)
    savetracetostream(stream, i*3, out, xcomp_rot)
    savetracetostream(stream, i*3+1, out, ycomp_rot)
    savetracetostream(stream, i*3+2, out, zcomp)
    
    if i%2000==0:
        print('Rotation of the input file in progress: {0:.1f}%'.format(i/(nbr_of_SP*nbr_of_geophone)*100))
    
out.stats = AttribDict()
#out.stats = Stats(dict(textual_file_header=stream.textual_file_header))
out.stats.textual_file_header = stream.textual_file_header
out.stats.binary_file_header = stream.binary_file_header
out.stats.binary_file_header.trace_sorting_code = stream.binary_file_header.trace_sorting_code  

print('Saving SEG-Y files...')
out.write('Horizontal_Rotation.sgy', format='SEGY', data_encoding=1, byteorder='big', textual_header_encoding='EBCDIC')


#xcomp = getfbdata(stream, 15000, 5, 10)
#ycomp = getfbdata(stream, 15001, 5, 10)
#zcomp = getfbdata(stream, 15002, 5, 10)
#angle = getangle(xcomp, ycomp, zcomp)
#xcomp_rot = np.require(xcomp * math.cos(math.radians(angle)) - ycomp * math.sin(math.radians(angle)), dtype=np.float32)
#ycomp_rot = np.require(xcomp * math.sin(math.radians(angle)) + ycomp * math.cos(math.radians(angle)), dtype=np.float32)
#plt.plot(xcomp, label='x')
#plt.plot(ycomp, label='y')
#plt.plot(zcomp, label='z')
#plt.grid(True)
#plt.legend()
#
#plt.plot(xcomp_rot, label='x rot')
#plt.plot(ycomp_rot, label='y_rot')
#plt.plot(zcomp, label='z')
#plt.grid(True)
#plt.legend()
#
#plt.plot(xcomp, ycomp)
#plt.axes().set_aspect('equal', 'datalim')
#plt.legend()
#
#plt.plot(xcomp_rot, ycomp_rot)
#plt.axes().set_aspect('equal', 'datalim')
#plt.legend()


#def FBTime(inputfile):
#    
#    # Read the input SEG-Y files, 3D-VSP stacked file, sorted in SP, MD, Trace Number (output of stack)
#    # Save it at a stream object, the seismic format from the Obspy library
#    
#    tic = timeit.default_timer()
#    print('The SEG-Y files is loading ... Please be patient')
#    stream = _read_segy(inputfile, headonly=True)
#    toc = timeit.default_timer()
#    print('The loading took {} seconds'.format(toc-tic))
#    
#    # Create a series of variable to be able to save each component of each geophone
#    # We will automatically figure out the following, number of geophone, first and last geophone's depth, number of shot points, number of components, the depth interval
#    
#    nbr_of_geophone = np.count_nonzero(np.unique(np.array([t.header.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group for t in stream.traces])))
#    
#    #first_geophone = np.array([t.header.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group for t in stream.traces]).min()/100
#    
#    #last_geophone = np.array([t.header.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group for t in stream.traces]).max()/100
#    
#    nbr_of_SP = np.count_nonzero(np.unique(np.array([t.header.energy_source_point_number for t in stream.traces])))
#    
#    nbr_of_components = np.count_nonzero(np.unique(np.array([t.header.trace_number_within_the_ensemble for t in stream.traces])))
#    
#    #MD_dz = (last_geophone - first_geophone)/(nbr_of_geophone-1)
#    
#    # We QC the variables by checking that the calculated number of traces is equal to the number of traces in the stream object (input SEG-Y file)
#    
#    nbr_of_traces = nbr_of_geophone * nbr_of_SP * nbr_of_components
#    
#    #nbr_of_samples_per_trace = stream.binary_file_header.number_of_samples_per_data_trace
#    
#    if nbr_of_traces == int(str(stream).split()[0]):
#        print('All Shot Points, Components and Geophones accounted for.')
#    
#    
#    # running_mean is a function with calculates the average in a moving window
#    def running_mean(x, N):
#        cumsum = np.cumsum(np.insert(x, 0, 0)) 
#        return (cumsum[N:] - cumsum[:-N]) / N 
#    
#    
#    tic = timeit.default_timer() #to measure the time it takes to calculate the first break picking
#    
#                            
#    #set up a series of variable for the automatic first break picker                          
#    first_arrival_time = np.empty(0, dtype=np.int32) #empty array to store the first arrival time
#    fore_window = 10 #10 samples in the fore window
#    back_window = int(fore_window * 1.5) #15 samples in the back window
#    
#    #the first ratio measurement is done at the end of the back window, therefore we need to add 'back_window - 1' sample to the ratio array to get the correct time of the maximum ratio sample
#    auto_pick_ratio_start = np.zeros(back_window - 1) 
#    
#    #set up a time array equal to 4 samples 
#    time = np.linspace(0, 3, 4) - 2
#    
#    #Start measuring first arrival time, source-receiver pair by source-receiver pair    
#    for i in range(nbr_of_SP*nbr_of_geophone):
#        #create a total vector trace by summing the absolute values of all three components for all samples
#        data_to_vectorise = np.sum([np.absolute(stream.traces[i*nbr_of_components + j].data) for j in range(nbr_of_components)], axis=0)
#
#        #measure the average value in each window on the total vector trace
#        mean_back_window = running_mean(data_to_vectorise, back_window)[:-fore_window]
#        mean_fore_window = running_mean(data_to_vectorise, fore_window)[back_window:]
#        
#        #calculate the ratio between fore and back window
#        auto_pick_ratio = mean_fore_window / mean_back_window 
#        
#        #add to the ratio array the 'back_window' samples with couldn't not be measured to have an array with an index equal to the time sample in the data
#        auto_pick_ratio = np.append(auto_pick_ratio_start, auto_pick_ratio)
#        
#        #extract the first arrival time precise to the nearest sample
#        rough_first_arrival_time = auto_pick_ratio.argmax()
#        
#        # interpolate around the rough first arrival time and find a better first break time
#        time_window = time + rough_first_arrival_time #time samples around the first arrival time -5/+4 samples
#        start = int(time_window[0]) #first time sample
#        end = int(time_window[-1]) #last time sample
#        amplitude_window = data_to_vectorise[start:end+1] #extract the amplitude of the total vector trace at these time samples
#        
#        #set up the interpolation function to measure amplitude at any time with a cubic spline function
#        f = interp1d(time_window, amplitude_window, kind='cubic')
#        
#        #interpolate every 0.1ms
#        interpolated_amp = np.array([f(t) for t in np.arange(start, end, 0.1)])
#        
#        #measure the average value in each window on the total vector trace
#        mean_back_window = running_mean(interpolated_amp, back_window)[:-fore_window]
#        mean_fore_window = running_mean(interpolated_amp, fore_window)[back_window:]
#        
#        #calculate the ratio between fore and back window
#        auto_pick_ratio = mean_fore_window / mean_back_window
#        ##### previous interpolation was running too slowly with 100 samples, went down to 40 samples, which now takes about 3ms (+5ms to read the data in the first place)
#        
#        #add the calculated first arrival time to the firt_arrival_time array n time (n = number of components)
#        first_arrival_time = np.append(first_arrival_time, np.repeat(auto_pick_ratio.argmax()/10 + start, nbr_of_components))
#        
#        #to save for future dev, this calculate the angle in degrees of the half right hodogram
#        #math.degrees(math.atan(curve_fit(lambda x, m: m*x, stream.traces[i*nbr_of_components].data[int(first_arrival_time[i*nbr_of_components])-fore_window:int(first_arrival_time[i*nbr_of_components])+fore_window], stream.traces[i*nbr_of_components].data[int(first_arrival_time[i*nbr_of_components])-fore_window:int(first_arrival_time[i*nbr_of_components])+fore_window])[0][0]))
#
#
#        #to save for plotting three components together with first arrival time
#        #x = 3*5000
#        #plt.plot(stream.traces[x].data,'r',stream.traces[x+1].data, 'b', stream.traces[x+2].data, 'g', first_arrival_time[x], first_arrival_time[x], 'ro', ms=10)
#
#        #every 2000 loops tell the users how much has been done of the calculation
#        if i%2000==0:
#            print('The automatic first break picking has done {0:.1f}% of the work, almost there!'.format(i/(nbr_of_SP*nbr_of_geophone)*100))
#    
#    tac = timeit.default_timer()
#    print('The automatic first break picking took {0:.1f} seconds'.format(tac-tic))
#    
#    # Save the first arrival time to a text file to import it later on
#    np.savetxt('FirstBreakTime.txt', first_arrival_time, fmt='%.2f')
#
#
#
##Write first arrival time to headers of input SEG-Y file
##
##for t in range(nbr_of_traces):
##    stream.traces[t].header.water_depth_at_group = int(first_arrival_time[t]*100)
##   
##                                                   
##output SEG-Y with the first break time in headers
##stream.write('FirstBreakPicked.sgy', data_encoding=5, endian='>')
# 
#
#def main(infile):
#    FBTime(infile)                                                    
#
#
#if __name__ == "__main__":
#    main(sys.argv[1])