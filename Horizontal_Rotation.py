Created on Mon May 15 12:28:11 2017

@author: Antoine Pare

This code takes a 3DVSP after first break picking and performs a statistical horizontal rotation
The horizontal rotation angle and the tool face are saved as an ascii file
usage: inputfile outputfile

"""

import sys

import math
import statsmodels.api as sm
import numpy as np
import pandas as pd
import array

from segpy.dataset import DelegatingDataset
from segpy.datatypes import LIMITS, PY_TYPES
from segpy.reader import create_reader
from segpy.writer import write_segy

from segpy_numpy.extract import extract_trace_headers
from segpy.trace_header import TraceHeaderRev1
from itertools import islice


class DimensionalityError(Exception):
    pass

class RotatedDataset(DelegatingDataset):

    def __init__(self, source_dataset, angle_to_rotate):
        super().__init__(source_dataset)
        self._limits = LIMITS[source_dataset.data_sample_format]
        self._py_type = PY_TYPES[source_dataset.data_sample_format]
        self._angle_to_rotate = angle_to_rotate

    #function to rotate the trace in function of the rotation angle
    def _rotation(self, trace_index):
        trace = np.array(self.source.trace_samples(trace_index))
        trace_num = self.source.trace_header(trace_index).ensemble_trace_num
        angle = self._angle_to_rotate[trace_index]
        
        if trace_num == 1:
            trace_p1 = np.array(self.source.trace_samples(trace_index+1))
            rotated_trace = trace * math.cos(math.radians(angle)) - trace_p1 * math.sin(math.radians(angle))
        elif trace_num == 2:
            trace_m1 = np.array(self.source.trace_samples(trace_index-1))
            rotated_trace = trace_m1 * math.sin(math.radians(angle)) + trace * math.cos(math.radians(angle))
        else:
            rotated_trace = trace
        
        # Ensure that we use a Python type compatible with the data sample format
        typed_trace = array.array('f', [self._py_type(sample) for sample in rotated_trace])

        # Clip to the range supported by the data sample format
        clipped_trace = array.array('f', [max(self._limits.min, min(self._limits.max, sample)) for sample in typed_trace])
        return clipped_trace

    def trace_samples(self, trace_index, start=None, stop=None):
        return self._rotation(trace_index)


#function to measure the angle between the x and y component
def _getangle(xcomp, ycomp, zcomp):
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

#function to calculate the tool face of each geophone and then the horizontal rotation angle
def _measure_rotation_angle(segy_reader):
    #extract the needed header values (every three traces)
    sp, vsp_md, fb, sour_x, sour_y, geo_x, geo_y = extract_trace_headers(
        reader=segy_reader,
        fields=(TraceHeaderRev1.energy_source_point_num,
                TraceHeaderRev1.water_depth_at_group,
                TraceHeaderRev1.datum_elevation_at_receiver_group,
                TraceHeaderRev1.source_x,
                TraceHeaderRev1.source_y,
                TraceHeaderRev1.group_x,
                TraceHeaderRev1.group_y),
        trace_indexes=islice(segy_reader.trace_indexes(), None, None, 3))
    
    #calculate the necessary attributes
    nbr_of_geophone = np.count_nonzero(np.unique(vsp_md))
    nbr_of_SP = np.count_nonzero(np.unique(sp))
    nbr_of_components = 3
    nbr_of_traces = nbr_of_geophone * nbr_of_SP * nbr_of_components
    sample_rate = int(segy_reader.trace_header(0).sample_interval/1000)
    
    #check if attributes match number of traces of the input file
    if nbr_of_traces == segy_reader.num_traces():
        print('All Shot Points, Components and Geophones accounted for.')
    else:
        nbr_of_traces = segy_reader.num_traces()
        print('Some traces are missing, please stop program and check SEG-Y file header')

    #calculate the azimuth of each shot point
    azimuth = np.round([(360 + math.degrees(
            math.atan2((sour_x[t] - geo_x[t]),
                       (sour_y[t] - geo_y[t])))) % 360 for t in range(sour_x.size)],
                                            decimals=2)
    
    horizontal_angles = np.array([])
    tool_face = np.array([])
    
    #calculate for each source-receiver pair of the input file the rotation angle
    for i,j,m in zip(range(int(nbr_of_traces/nbr_of_components)), azimuth, fb):
        start = int(m/100/sample_rate) -5
        stop = int(m/100/sample_rate) +10
       
        xcomp = np.array(segy_reader.trace_samples(i*3, start, stop))
        ycomp = np.array(segy_reader.trace_samples(i*3 + 1, start, stop))
        zcomp = np.array(segy_reader.trace_samples(i*3 + 2, start, stop))
        
        horizontal_angles_i = _getangle(xcomp, ycomp, zcomp)
        horizontal_angles = np.append(horizontal_angles, horizontal_angles_i)
        
        tool_face_i = j - horizontal_angles_i
        if tool_face_i < 0:
            tool_face_i = tool_face_i + 360
        elif tool_face_i > 360:
            tool_face_i = tool_face_i - 360
        tool_face = np.append(tool_face, tool_face_i)
        
        if i%2000==0:
            print('The horizontal rotation calculation is in progress: {0:.1f}%'.format(i/(nbr_of_traces/nbr_of_components)*100))
            
    #bring all results in a dataframe to calculate the tool face (average every two degrees)        
    table_angle = np.column_stack((vsp_md, azimuth, horizontal_angles, tool_face))
    df = pd.DataFrame(table_angle, columns=['MD', 'shot azimuth', 'rotation angle', 'tool face'])
    averaged_tool_face = df.groupby('MD')['tool face'].agg(lambda x: np.histogram(x, bins=180,range=(0,360))[0].argmax()).values * 2
    averaged_tool_face = np.tile(averaged_tool_face, nbr_of_SP)
    angle_to_rotate = np.repeat(azimuth - averaged_tool_face, nbr_of_components)
    averaged_tool_face = np.repeat(averaged_tool_face, nbr_of_components)
    
    return angle_to_rotate, averaged_tool_face


def transform(in_filename, out_filename):
    with open(in_filename, 'rb') as in_file, \
         open(out_filename, 'wb') as out_file:
        
        print('Loading SEG-Y file')
        segy_reader = create_reader(in_file)
        print('Calculation starts...')
        angle_to_rotate, averaged_tool_face = _measure_rotation_angle(segy_reader)
        np.savetxt(in_filename[:-4]+'_RotationAngle.txt', angle_to_rotate, fmt='%.2f')
        np.savetxt(in_filename[:-4]+'_ToolFace.txt', averaged_tool_face, fmt='%.2f')

        transformed_dataset = RotatedDataset(segy_reader, angle_to_rotate)
        print('Rotating and saving SEG-Y file')
        write_segy(out_file, transformed_dataset)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    in_filename = argv[0]
    out_filename = argv[1]

    transform(in_filename, out_filename)

if __name__ == '__main__':
    sys.exit(main())
