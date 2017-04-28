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
