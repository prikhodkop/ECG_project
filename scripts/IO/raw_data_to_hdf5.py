import pandas as pd
import tables
import os, sys, time
import numpy as np

class Person(tables.IsDescription):
    gidn               = tables.UInt64Col()
    measurement_start  = tables.UInt64Col()
    
def create_hdf5(fileh):    
    # Create an Array with variable length of rows:
    pulse_peaks = fileh.create_vlarray(
                fileh.root,
                'pulses',
                tables.Float64Atom(shape=()), 
                "ragged array of floats",
                filters=tables.Filters(1),
                expectedrows = 100000)
 
    # Create a table with gidns and measurment starts
    pulse_types = fileh.create_vlarray(
                fileh.root, 
                'pulses_types',
                tables.StringAtom(itemsize = 1, shape=()),
                "ragged array of char",
                filters=tables.Filters(1),
                expectedrows = 100000)

    persons = fileh.create_table(
                fileh.root, 
                'persons', 
                Person, 
                "Readout example",
                filters=tables.Filters(1),
                expectedrows = 2000)
    return (pulse_peaks, pulse_types, persons)

def data_to_hdf5(dirpath_in, file_out):
    allfiles = os.listdir(dirpath_in)
    
    #import pdb; pdb.set_trace()
    
    fileh = tables.open_file(file_out, mode='w')
    
    pulses_peaks, pulses_types, persons = create_hdf5(fileh)
    
    for i, filename in enumerate(allfiles):
        gidn = int(filename.split('.')[0])
        filepath  = os.path.join(dirpath_in, filename)
        with open(filepath) as timereader:
            timestr = timereader.readline().rstrip('\r\n')
            hour, minute, second = timestr.split(':')
            measurement_start = ((int(hour) * 60 + int(minute)) * 60 + int(second)) * 1000
            pulse_peaks = pd.read_csv(filepath, 
                    sep      = '\t',
                    skiprows = 1,
                    header   = None, 
                    names    = ('start', 'delay', 'peak_type'))
            
            pulse_peaks = pulse_peaks[pulse_peaks.start != 0]
            pulse_peaks = pulse_peaks[pulse_peaks.start != pulse_peaks.delay]
            pulse_peaks = pulse_peaks[pulse_peaks.peak_type == 'N']
            pulse_peaks = pulse_peaks[pulse_peaks.delay > 0]
            pulse_peaks = pulse_peaks[pulse_peaks.delay < 5000]

            
            # Append rows:
            person = persons.row
            person['gidn']  = gidn
            person['measurement_start'] = measurement_start
            # Insert a new particle record
            person.append()
            pulses_peaks.append(np.array(pulse_peaks.start, dtype = np.float64))
            pulses_types.append(np.array(pulse_peaks.peak_type, dtype = 'S1'))
            if i%10 == 0:
                print '\rFinished %d' % i,
                sys.stdout.flush()
                time.sleep(0.005)
    print '\rFinished %d' % i
    # Close the file.
    fileh.close()

# function get path to folder with .RR files and path and name new h5 file
def data_to_hdf5 (path_to_RR, path_to_h5):
# path_to_RR = 'rawdata/data'
# path_to_h5 = 'compute_data/all_pulses_only_N_for_exp.h5'
    data_to_hdf5(path_to_RR, path_to_h5)


