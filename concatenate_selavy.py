from astropy.table import Table,vstack,Column
import pandas
import glob
import re
import os
import sys
import argparse

directory = sys.argv[1]

filetype='components'
filelist=os.path.join(directory,'*{}.txt'.format(filetype))
selavy_files=sorted(glob.glob(filelist))
data=[]
for i,filename in enumerate(selavy_files):    
    field=os.path.splitext(os.path.split(filename)[-1])[0]
    if i==0:
        if 'EPOCH' in filename:
            m=re.match(r'.*EPOCH(\w+?)\..*',filename)
            epoch=m.groups()[0]
        else:
            epoch='0'
        path = os.path.split(filename)[0]
        stokes = os.path.split(filename)[-1].split('.')[2]
    data.append(Table.from_pandas(pandas.read_fwf(filename, skiprows=[1,])))
    data[-1].add_column(Column([field]*len(data[-1]),
                               name='field'))
    if field.startswith('n'):
        # negative components for Stokes V
        data[-1]['flux_peak']*=-1
        data[-1]['flux_peak_err']*=-1
        data[-1]['flux_int']*=-1
        data[-1]['flux_int_err']*=-1
    print('Read {0} lines from {1} ({2}/{3})'.format(len(data[-1]),
                                                     filename,i,len(selavy_files)))
    
vast_data=vstack(data)
# add some units
units={'ra_deg_cont': 'deg',
       'dec_deg_cont': 'deg',
       'flux_peak': 'mJy/beam',
       'flux_peak_err': 'mJy/beam',
       'flux_int': 'mJy',
       'flux_int_err': 'mJy',
       'maj_axis': 'arcsec',
       'maj_axis_err': 'arcsec',
       'min_axis': 'arcsec',
       'min_axis_err': 'arcsec',
       'pos_ang': 'deg',
       'pos_ang_err': 'deg',
       'maj_axis_deconv': 'arcsec',
       'maj_axis_deconv_err': 'arcsec',
       'min_axis_deconv': 'arcsec',
       'min_axis_deconv_err': 'arcsec',
       'pos_ang_deconv': 'deg',
       'pos_ang_deconv_err': 'deg',
       'ra_err': 'arcsec',
       'dec_err': 'arcsec',
       'rms_fit_gauss': 'mJy/beam',
       'rms_image': 'mJy/beam',
       }
for column,unit in units.items():
    vast_data[column].unit=unit

# these are empty and confuse HDF5
del vast_data['#']
del vast_data['comment']
vast_data.meta={'epoch': epoch}
vast_data.write(os.path.join(path,'VAST_Epoch{}_Stokes{}_{}.hdf5'.format(epoch,stokes,filetype)),path='data',overwrite=True,serialize_meta=True)
print('Wrote to %s' % os.path.join(path,'VAST_Epoch{}_Stokes{}_{}.hdf5'.format(epoch,stokes,filetype)))

