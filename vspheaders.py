

def set_VSP_headers():
    
    import obspy
    
    import obspy.io.segy.header    
    
    obspy.io.segy.header.TRACE_HEADER_FORMAT[:] = [
        # [length, name, special_type, start_byte]
        # Special type enforces a different format while unpacking using struct.
        [4, 'trace_sequence_number_within_line', False, 0],
        [4, 'trace_sequence_number_within_segy_file', False, 4],
        [4, 'field_file_number', False, 8],
        [4, 'field_chanel_number', False, 12],
        [4, 'shotpoint_number', False, 16],
        [4, 'ident_number', False, 20],
        [4, 'trace_number', False, 24],
        [2, 'data_trace_type', False, 28],
        [2, 'number_of_vertically_summed_traces_yielding_this_trace', False, 30],
        [2, 'number_of_horizontally_stacked_traces_yielding_this_trace', False,
         32],
        [2, 'data_use', False, 34],
        [4, 'source_detect_distance' +
         'the_center_of_the_receiver_group', False, 36],
        [4, 'elevation_detect', False, 40],
        [4, 'elevation_source', False, 44],
        [4, 'depth_source', False, 48],
        [4, 'vsp_measured_depth', False, 52],
        [4, 'depth_from_datum', False, 56],
        [4, 'depth_detected', False, 60],
        [4, 'first_break_time', False, 64],
        [2, 'scalar_to_be_applied_to_all_elevations_and_depths', False, 68],
        [2, 'scalar_to_be_applied_to_all_coordinates', False, 70],
        [4, 'x_cord_source', False, 72],
        [4, 'y_cord_source', False, 76],
        [4, 'x_cord_detect', False, 80],
        [4, 'y_cord_detect', False, 84],
        [2, 't_gain_exponant', False, 88],
        [2, 'weathering_velocity', False, 90],
        [4, 'trace_balance_factor', False, 92],
        [4, 'horizontal_rotation_angle', False, 96],
        [4, 'vertical_rotation_angle', False, 100],
        [2, 'lag_time_A', False, 104],
        [2, 'lag_time_B', False, 106],
        [4, 'radial_comp_azimuth', False, 108],
        [2, 'mute_time_end_time_in_ms', False, 112],
        [2, 'number_of_samples_in_this_trace', 'H', 114],
        [2, 'sample_interval_in_ms_for_this_trace', 'H', 116],
        [2, 'gain_type_of_field_instruments', False, 118],
        [4, 'borehole_azimuth', False, 120],
        [4, 'tool_azimuth', False, 124],
        [2, 'sweep_frequency_at_end', False, 128],
        [2, 'mlr_tool_number', False, 130],
        [2, 'sweep_type', False, 132],
        [2, 'sweep_trace_taper_length_at_start_in_ms', False, 134],
        [2, 'sweep_trace_taper_length_at_end_in_ms', False, 136],
        [2, 'taper_type', False, 138],
        [2, 'alias_filter_frequency', False, 140],
        [2, 'alias_filter_slope', False, 142],
        [2, 'notch_filter_frequency', False, 144],
        [2, 'notch_filter_slope', False, 146],
        [2, 'low_cut_frequency', False, 148],
        [2, 'high_cut_frequency', False, 150],
        [2, 'low_cut_slope', False, 152],
        [2, 'high_cut_slope', False, 154],
        [2, 'year_data_recorded', False, 156],
        [2, 'day_of_year', False, 158],
        [2, 'hour_of_day', False, 160],
        [2, 'minute_of_hour', False, 162],
        [2, 'second_of_minute', False, 164],
        [2, 'time_basis_code', False, 166],
        [2, 'trace_weighting_factor', False, 168],
        [2, 'geophone_group_number_of_roll_switch_position_one', False, 170],
        [4, 'static_correction_source', False, 172],
        [2, 'gap_size', False, 176],
        [2, 'over_travel_associated_with_taper', False, 178],
        [4, 'x_coordinate_of_ensemble_position_of_this_trace', False, 180],
        [4, 'y_coordinate_of_ensemble_position_of_this_trace', False, 184],
        [4, 'for_3d_poststack_data_this_field_is_for_in_line_number', False, 188],
        [4, 'borehole_deviation', False,
         192],
        [4, 'shotpoint_number_other', False, 196],
        [4, 'station_number_source', False, 200],
        [4, 'time_shit_alignment', False, 204],
        [2, 'transduction_constant_exponent', False, 208],
        [2, 'transduction_units', False, 210],
        [2, 'device_trace_identifier', False, 212],
        [2, 'scalar_to_be_applied_to_times', False, 214],
        [2, 'source_type_orientation', False, 216],
        [4, 'source_energy_direction_mantissa', False, 218],
        [2, 'source_energy_direction_exponent', False, 222],
        [4, 'source_measurement_mantissa', False, 224],
        [2, 'source_measurement_exponent', False, 228],
        [2, 'source_measurement_unit', False, 230],
        [8, 'unassigned', False, 232]]
    
    
    obspy.io.segy.header.TRACE_HEADER_KEYS[:] = [_i[1] for _i in obspy.io.segy.header.TRACE_HEADER_FORMAT]