"""
Written by Antoine Paré in April 2017

This code takes a 3D-VSP after first break picking and perform a statistical horizontal rotation

"""
import sys
import timeit
import numpy as np
import math
import statsmodels.api as sm
import pandas as pd

from segpy.reader import create_reader
from segpy.writer import write_segy

def getfbdata(segy_reader, idx, window_min, window_max):
    """
    function that extracts data around the FB (First Break) of SEG-Y file (segy_reader) at trace idx, 
    from window_min samples before FB to window_max samples after FB
    """
    fb = int(segy_reader.trace_header(idx).water_depth_at_group/100)
    return segy_reader.trace_samples(idx, fb - int(window_min), fb + (window_max) + 1)

def getangle(xcomp, ycomp, zcomp):
    """
    this function inputs the three component data around the FB and calculates the rotation 
    angle to maximise the energy on the Y component with the same polarity as the Z component
    """
    model = sm.OLS(xcomp, ycomp) #Model to measure the gradient of the crossplot between X and Y
    results = model.fit() #Measures the gradient
    angle = round((360 + math.degrees(math.atan2(results.params[0],1))) % 360, 2) #Transforms the gradient into an azimuth from North
    ycomp_rot = xcomp * math.sin(math.radians(angle)) + ycomp * math.cos(math.radians(angle)) #Check polarity of the rotated Y component
    if np.sign(ycomp_rot)[7] != np.sign(zcomp)[7]: #If the polarity of the rotated Y comp is different from Z, add 180 to angle
        if angle <= 180:
            angle = angle + 180
        else:
            angle = angle - 180
    return angle

inputfile = open('test.sgy', 'rb')

print('The SEG-Y files is loading ... Please be patient')
tic = timeit.default_timer()
segy_reader = create_reader(inputfile)
toc = timeit.default_timer()
print('The loading took {0:.0f} seconds'.format(toc-tic))

""" extract all source and receiver (geophone) coordinates """
sour_x = np.array([segy_reader.trace_header(trace_index).source_x for trace_index in segy_reader.trace_indexes()])[::3]
sour_y = np.array([segy_reader.trace_header(trace_index).source_y for trace_index in segy_reader.trace_indexes()])[::3]
geo_x = np.array([segy_reader.trace_header(trace_index).group_x for trace_index in segy_reader.trace_indexes()])[::3]
geo_y = np.array([segy_reader.trace_header(trace_index).group_y for trace_index in segy_reader.trace_indexes()])[::3]

""" measures the source to receiver azimuth """
azimuth = np.round([(360 + math.degrees(math.atan2((sour_x[t] - geo_x[t]), (sour_y[t] - geo_y[t])))) % 360 for t in range(sour_x.size)], decimals=2)

""" create a list of all the geophones' depth """
geophone_depths = (np.array([segy_reader.trace_header(trace_index).datum_elevation_at_receiver_group for trace_index in segy_reader.trace_indexes()])/100)[::3]                       
  
""" Calculate the number of tools, shot points and components in the 3D-VSP, then check that the calculated number of traces 
is equal to the number of traces in the SEG-Y file """    
nbr_of_geophone = np.count_nonzero(np.unique(geophone_depths))
nbr_of_SP = np.count_nonzero(np.unique(np.array([segy_reader.trace_header(trace_index).energy_source_point_num for trace_index in segy_reader.trace_indexes()])))
nbr_of_components = np.count_nonzero(np.unique(np.array([segy_reader.trace_header(trace_index).trace_num for trace_index in segy_reader.trace_indexes()])))
nbr_of_traces = nbr_of_geophone * nbr_of_SP * nbr_of_components

if nbr_of_traces == segy_reader.num_traces():
    print('All Shot Points, Components and Geophones accounted for.')
else:
    print('Some traces are missing, please stop program and check SEG-Y file header')

""" set up empty arrays for the output horizontal angle of rotation and the tool face """
horizontal_angles = np.array([])
tool_face = np.array([])

""" calculate for all source-receiver pairs the horizontal angle of rotation and then the tool face """
tic = timeit.default_timer()

for i,j in zip(range(nbr_of_SP*nbr_of_geophone), azimuth):
    xcomp = getfbdata(segy_reader, i*3, 5, 10)
    ycomp = getfbdata(segy_reader, i*3 +1, 5, 10)
    zcomp = getfbdata(segy_reader, i*3 +2, 5, 10)
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

""" Calculate the average measured tool face for each tool """
#TO ADD need to add a better measurement of the angle to rotate, not using .mean() but using numpy.histogram
angle_to_rotate = azimuth - np.tile(df.groupby('MD')['tool face'].mean().values(), nbr_of_SP)

inputfile.close()
