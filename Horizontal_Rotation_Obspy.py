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


def getfbdata(stream, idx, window_min, window_max):
    """
    function that extracts data around the FB (First Break) of SEG-Y file (segy_reader) at trace idx, 
    from window_min samples before FB to window_max samples after FB
    """
    fb = int(stream.traces[idx].header.water_depth_at_group/100)
    return stream.traces[idx].data[(fb - int(window_min)) : (fb + (window_max) + 1)]


def getangle(xcomp, ycomp, zcomp):
    """
    this function inputs the three component data around the FB and calculates the rotation 
    angle to maximise the energy on the Y component with the same polarity as the Z component
    """
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

    

inputfile = 'test.sgy'

print('The SEG-Y files is loading ... Please be patient')
tic = timeit.default_timer()
stream = _read_segy(inputfile, headonly=True)
toc = timeit.default_timer()
print('The loading took {0:.0f} seconds'.format(toc-tic))


""" extract all source and receiver (geophone) coordinates """
source_x = np.array([t.header.source_coordinate_x for t in stream.traces])[::3]
source_y = np.array([t.header.source_coordinate_y for t in stream.traces])[::3]
geo_x = np.array([t.header.group_coordinate_x for t in stream.traces])[::3]
geo_y = np.array([t.header.group_coordinate_x for t in stream.traces])[::3]


""" measures the source to receiver azimuth """
azimuth = np.round([(360 + math.degrees(math.atan2((source_x[t] - geo_x[t]), (source_y[t] - geo_y[t])))) % 360 for t in range(source_x.size)], decimals=2)


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

    
""" set up empty arrays for the output horizontal angle of rotation and the tool face """
horizontal_angles = np.array([])
tool_face = np.array([])


""" calculate for all source-receiver pairs the horizontal angle of rotation and then the tool face """
tic = timeit.default_timer()
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

toc = timeit.default_timer()
print('The calculation took {0:.0f} seconds'.format(toc-tic))

""" Sum up the results in a data frame """        
table_angle = np.column_stack((geophone_depths, azimuth, horizontal_angles, tool_face))
df = pd.DataFrame(table_angle, columns=['MD', 'shot azimuth', 'rotation angle', 'tool face'])

""" Calculate the average measured tool face for each tool and use it to determine the angle to rotate 
    each source - receiver pair (angle_to_rotate) """
#need to add a better measurement of the angle to rotate, not using .mean() but using numpy.histogram
angle_to_rotate = azimuth - np.tile(df.groupby('MD')['tool face'].mean().values, nbr_of_SP)


""" Set up an empty stream which will hold the rotated traces to be saved as a SEGY """
out = Stream()

""" Rotate all the traces in stream using the horizontal rotation angle and save them in the output stream: out """
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

""" set up the necessary textual and binary file header for the output stream: out """
out.stats = AttribDict()
#out.stats = Stats(dict(textual_file_header=stream.textual_file_header))
out.stats.textual_file_header = stream.textual_file_header
out.stats.binary_file_header = stream.binary_file_header
out.stats.binary_file_header.trace_sorting_code = stream.binary_file_header.trace_sorting_code  

""" save 'out' as a SEG-Y file """
print('Saving SEG-Y files...')
out.write('Horizontal_Rotation.sgy', format='SEGY', data_encoding=1, byteorder='big', textual_header_encoding='EBCDIC')
