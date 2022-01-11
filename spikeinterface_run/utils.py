import numpy as np
import pandas as pd
import sys,os
import pylab
from scipy import stats,signal,io
import matplotlib.pyplot as plt

import pickle
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
import h5py


import matplotlib
matplotlib.use('pdf')

plt.rcParams['pdf.fonttype'] = 'truetype'

from functools import reduce
import datetime

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
from spikeinterface.exporters import export_to_phy, export_report


import argparse
from subprocess import call
import time

from glob import glob

from probeinterface import get_probe, read_prb



def run_sorting(input_path):

  # input_path e.g. /n/groups/datta/guitchounts/data/gmou51/gmou51_2021-12-15_18-08-21_odor/Record Node 101/ 

  ## remove any pesky backslashes in path:
  input_path = input_path.replace('\\','')

  data = se.read_openephys(input_path)
  probe = read_prb(glob('%s/../../*.prb' % input_path)[0] ).probes[0]    #('/home/gg121/code/spikeinterface_analysis/A4x16-Poly3-5mm-20-200-160-H64LP.prb').probes[0]
  recording = data.set_probe(probe)



  print('Channel ids:', recording.get_channel_ids())
  
  
  recording = st.preprocessing.notch_filter(recording,freq=60)

  recording_f = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
  recording_cmr = st.preprocessing.common_reference(recording_f, reference='global',  operator='median')
  
  

  ## Load and set parameters for sorting:

  default_ms4_params = ss.Mountainsort4Sorter.default_params()
  num_workers = 8
  ms4_params = default_ms4_params.copy()
  ms4_params['adjacency_radius'] = 50
  ms4_params['detect_sign'] = 0
  ms4_params['filter'] = False
  ms4_params['num_workers'] = num_workers

  ms4_params['clip_size'] = 64 # waveform is 64 samples
  ms4_params['detect_interval'] = 15 # 0.5 ms (?)

  fs = 3e4



  #### run sorting:
  print('Starting mountainsort4 sorting....')
  start = time.time()
  # run Mountainsort:

  sorting_path = '%s/../tmp_MS4' % input_path
  firings_path = '%s/../tmp_MS4/firings.npz' % input_path

  if not os.path.exists(firings_path):

    sorting_MS4 = ss.run_sorter('mountainsort4',recording_cmr,  # parallel=True,
                           verbose=True,
                           output_folder=sorting_path, **ms4_params)

    print('sorting finished in %f seconds...' % (time.time() - start) )

  else:
    print('Loading pre-computed sorting')
    sorting_MS4 = si.core.NpzSortingExtractor(firings_path)

  ### Extract waveforms, compute pcs, export to phy, and save sorting report:
  
  print('Extracting waveforms')
  we = si.extract_waveforms(recording_cmr, sorting_MS4, '%s/tmp_MS4/waveforms' % input_path,
            ms_before=1,ms_after=1,load_if_exists=True,
                           n_jobs=8, total_memory='1G') # 


  print('Computing PCs')
  pc = st.compute_principal_components(we, load_if_exists=True,
                                       n_components=3)


  print('Computing Metrics')
  qc = st.compute_quality_metrics(we, waveform_principal_component=pc)

  qc.to_csv('%s/tmp_MS4/metrics.csv' % input_path)

  
  good_units = np.intersect1d(np.where(qc['snr'] >= 3.5), np.where(qc['isi_violations_rate'] < 0.2))
  #np.intersect1d(np.where(metrics['isi_violation'].values <= 0.015), np.where(metrics['snr'].values >=3.5 ) )

  
  all_unit_ids = sorting_MS4.get_unit_ids()
  np.savez('%s/tmp_MS4/unit_properties' % input_path,all_unit_ids=all_unit_ids,good_units=good_units)

  print('saved unit properties')

  sorting_MS4.dump_to_pickle('%s/tmp_MS4/sorting.pickle' % input_path)
  print('saved unit sorting pickle')



  ##### PLOT 
  ### get mapping of channel id to location:
  channel_ids = recording.get_channel_ids()
  channel_locations = recording.get_channel_locations()
  data_channels = {}
  for i in range(64):
      data_channels[channel_ids[i]] = channel_locations[i]


  for unit in good_units:

      waveform = we.get_waveforms(unit_id=unit)

      f = plt.figure(dpi=600,figsize=(4,4))
      for ch,key in enumerate(data_channels):
      #for key in range(64):
          plt.plot(np.arange(0,20,20/60) + data_channels[key][0],
                  waveform[:,:,ch].mean(axis=0) + data_channels[key][1], 
                  color='k', lw=0.25)

      plt.title('unit %d, snr=%.2f, ISI=%.2f' % (unit,metrics['snr'].values[unit],
                                                 metrics['isi_violation'].values[unit] ) )
      sns.despine(left=True,bottom=True)
      
      f.savefig('%s/tmp_MS4/unit_%d.pdf' % (input_path,unit) )
      
      plt.close(f)




  


  try:
    print('Exporting to Phy')
    export_to_phy(we, '%s/tmp_MS4/phy' % input_path , 
                n_jobs=8, total_memory='1G',
                #peak_sign='neg',
                copy_binary=False,
                progress_bar=True, #verbose=True,
              #remove_if_exists=True,
                )

  except:
    print('Export to phy failed')


  try:
    print('Exporting spike sorting report')
    export_report(we, '%s/tmp_MS4/report' % input_path )
  except:
    print('Export report failed')

