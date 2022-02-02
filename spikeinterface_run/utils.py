import numpy as np
import pandas as pd
import sys,os
import pylab
from scipy import stats,signal,io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
import h5py


import matplotlib
matplotlib.use('pdf')

plt.rcParams['pdf.fonttype'] = 'truetype'

from functools import reduce
from datetime import datetime

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



def run_sorting(input_path,sorter_path = 'tmp_MS4'):

  fs = 3e4


  ## if this is a triple-underscore-separated concatenated string, separate it; otherwise make it a list of one:
  if len(input_path.split('___'))>1:
    input_path = input_path.split('___')
  else:
    input_path = [input_path]
  
  print('INPUT PATH in run_sorting = ', input_path)

  recordings_list = []
  recordings_length_idxs = [0] ## list of numbers of frames in each recording...

  for sesh_idx, session in enumerate(input_path):
      
      

      data = se.read_openephys(glob('%s/Record*' % session)[0])

      ## remove the Record Node from input
      if 'Record' in session:
        session = os.path.split(session)[0] 


      probe = read_prb(glob('%s/../*.prb' % session)[0] ).probes[0]
      recording = data.set_probe(probe)


      recording = st.preprocessing.notch_filter(recording,freq=60)
      recording_f = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
      recording_cmr = st.preprocessing.common_reference(recording_f, reference='global',  operator='median')
   
      # # shorten the recording for testing
      #recording_cmr = recording_cmr.frame_slice(start_frame=0*fs, end_frame=5*fs)



      recordings_list.append(recording_cmr)
      
      recordings_length_idxs.append(recordings_length_idxs[sesh_idx] + recording_cmr.get_num_frames()) # recording_cmr.get_num_samples() instead? same thing?
      

  multirecording = si.concatenate_recordings(recordings_list) 


  # find the base path - the common mouse folder here
  base_paths = np.array([os.path.split(session)[0] for session in input_path])
  base_path = base_paths[0] if np.all(base_paths == base_paths) else None

  
  dates = np.array([session.split('_')[1] for session in input_path])
  date = dates[0] if np.all(dates == dates) else None
  
  sorter_full_path = '%s/concat_sorting/%s/' % (base_path,date)
  if not os.path.exists(sorter_full_path):
      os.makedirs(sorter_full_path, exist_ok=True)

  firings_path = '%s/firings.npz' % sorter_full_path


  
  

  ## Load and set parameters for sorting:

  default_ms4_params = ss.Mountainsort4Sorter.default_params()
  num_workers = int(os.environ['SLURM_JOB_CPUS_PER_NODE']) # 8
  ms4_params = default_ms4_params.copy()
  ms4_params['adjacency_radius'] = 50
  ms4_params['detect_sign'] = -1
  ms4_params['filter'] = False
  ms4_params['num_workers'] = num_workers

  ms4_params['clip_size'] = 64 # waveform is 64 samples
  ms4_params['detect_interval'] = 15 # 0.5 ms (?)

  



  #### run sorting:
  
  # run Mountainsort:

  #sorter_full_path = '%s/%s' % (input_path, sorter_path)
  

  if not os.path.exists(firings_path):
    print('Starting mountainsort4 sorting....')
    start = time.time()
    
    sorting_MS4 = ss.run_sorter('mountainsort4',multirecording,  # parallel=True,
                           verbose=True,
                           output_folder=sorter_full_path, **ms4_params)

    print('sorting finished in %f seconds...' % (time.time() - start) )


  else:
    print('Loading pre-computed sorting')
    sorting_MS4 = si.core.NpzSortingExtractor(firings_path)



  #################################### save spike trains in folders of individual sessions: ####################################
  
  tmp_spike_trains = sorting_MS4.get_all_spike_trains()[0]
  tmp_spike_times = tmp_spike_trains[0]
  tmp_spike_labels = tmp_spike_trains[1]


  for sesh_idx, session in enumerate(input_path):
    
    
    start,stop = recordings_length_idxs[sesh_idx],recordings_length_idxs[sesh_idx+1]
    
    tmp_spike_idx = (tmp_spike_trains[0]>=start) & (tmp_spike_trains[0] < stop) 
    
    session_spike_times = tmp_spike_times[tmp_spike_idx] - tmp_spike_times[tmp_spike_idx][0] # normalize so it starts at 0 again.
    session_spike_labels = tmp_spike_labels[tmp_spike_idx]
    
    
    session_local_path = os.path.split(session)[1]
    session_firings_save_path = '%s/%s/%s/' % (base_path,session_local_path,sorter_path )
    if not os.path.exists(session_firings_save_path):
         os.makedirs(session_firings_save_path, exist_ok=True)
    
    np.savez('%s/firings.npz' % session_firings_save_path,
             unit_ids=sorting_MS4.get_unit_ids(),
             spike_indexes = session_spike_times, spike_labels =  session_spike_labels)
    
    print(session,start,stop,session_spike_times[-1])
    





  ### Extract waveforms, compute pcs, export to phy, and save sorting report:
  
  print('Extracting waveforms')
  ms_before=1
  ms_after=1
  waveform_time = ms_before + ms_after # 2 ms, 60 timepoints
  we = si.extract_waveforms(multirecording, sorting_MS4, '%s/waveforms' % sorter_full_path,
            ms_before=ms_before,ms_after=ms_after,load_if_exists=True,
                           n_jobs=num_workers, total_memory='1G') # 


  
  print('Computing PCs')
  pc = st.compute_principal_components(we, load_if_exists=True,
                                       n_components=3)


  print('Computing Metrics')
  qc_path = '%s/metrics.csv' % sorter_full_path
  if not os.path.exists(qc_path):
    qc = st.compute_quality_metrics(we, waveform_principal_component=pc) # load_if_exists=True -- not an option in 0.91
    qc.to_csv(qc_path)
  else:
    qc = pd.read_csv(qc_path,index_col='Unnamed: 0')

  
  # change--now saving unit_ids rather than the index of list of all units.
  good_units = qc[(qc['snr'] >=3.5) & (qc['snr'] <=25) & (qc['isi_violations_rate'] < 0.2) & (qc['firing_rate'] >= 0.1)  & (qc['presence_ratio'] >= 0.5)   ].index


  
  all_unit_ids = sorting_MS4.get_unit_ids()
  np.savez('%s/unit_properties' % sorter_full_path,all_unit_ids=all_unit_ids,good_units=good_units)

  print('saved unit properties')

  sorting_MS4.dump_to_pickle('%s/sorting.pickle' % sorter_full_path)
  print('saved unit sorting pickle')



  ##### PLOT 
  ### get mapping of channel id to location:
  channel_ids = recording.get_channel_ids()
  channel_locations = recording.get_channel_locations()
  data_channels = {}
  for i in range(64):
      data_channels[channel_ids[i]] = channel_locations[i]


  ## save figs for all units, but save the good units also in a special palce:
  good_unit_plot_path = '%s/good_units/' % sorter_full_path
  if not os.path.exists(good_unit_plot_path):
      os.makedirs(good_unit_plot_path, exist_ok=True)


  
  waveform_scale_factor = 0.1 # how much to downscale waveforms in the plot
  plot_x_lim = 750
  plot_y_lim = 150

  for unit in sorting_MS4.get_unit_ids(): 

      waveform = we.get_waveforms(unit_id=unit)

      f = plt.figure(dpi=600,figsize=(4,4))
      ax = plt.subplot(111)
      
      for ch,key in enumerate(data_channels):
      

          ax.plot(np.arange(0,20,20/(fs*waveform_time / 1000)) + probe._contact_positions[ch,0],
                waveform_scale_factor * waveform[:,:,ch].mean(axis=0) + probe._contact_positions[ch,1], 
                color='k', lw=0.25)
      
      # in x, 20 microns represents 60 time steps (2 ms)... 
      ax.plot([plot_x_lim - 20, plot_x_lim],[0,0],c='k',lw=0.5 ) # time scale bar
      scale_bar_uv = 200
      ax.plot([plot_x_lim, plot_x_lim],[0, waveform_scale_factor * scale_bar_uv  ],c='k',lw=0.5 ) # µV scale bar
      ax.text(plot_x_lim,-10,'%d ms' % waveform_time, fontsize=6)
      ax.text(plot_x_lim+10,10,'%d µV' % scale_bar_uv,rotation=90, fontsize=6)

      
      ax.set_title('unit %d, snr=%.2f, ISI=%.2f' % (unit,qc.loc[unit]['snr'],
                                               qc.loc[unit]['isi_violations_rate'] ) )

      ax.set_ylim([0,plot_y_lim])
      ax.set_xlim([0,plot_x_lim])
      ax.set_ylabel('Depth (µm)')
      ax.set_xlabel('Shank Distance (µm)')

      divider = make_axes_locatable(ax)
      cax = divider.append_axes('bottom', size='50%', pad=0.05)
      
      
      
      metrics_for_plot = qc.loc[unit][:5] 

      table = cax.table(cellText=metrics_for_plot.apply('{:,.2f}'.format).values.reshape(-1,1).T,
                colLabels=metrics_for_plot.T.index, loc='center',edges='open',)
      table.auto_set_font_size(False)
      table.set_fontsize(3)

      cax.set_yticks([])
      cax.set_xticks([])
      sns.despine(ax=cax,left=True,bottom=True)
      
      cax.patch.set_alpha(0)
      
      sns.despine(offset=5)

      f.savefig('%s/unit_%d.pdf' % (sorter_full_path,unit))

      if unit in good_units:
        f.savefig('%s/unit_%d.pdf' % (good_unit_plot_path,unit))

      plt.close(f)




  try:
    print('Exporting to Phy')
    export_to_phy(we, '%s/phy' % sorter_full_path, 
                n_jobs=num_workers, total_memory='1G',
                #peak_sign='neg',
                copy_binary=False,
                progress_bar=True, #verbose=True,
              #remove_if_exists=True,
                )

  except Exception as e:
    print('Export to phy failed')
    print(e)


  # try:
  #   print('Exporting spike sorting report')
  #   export_report(we, '%s/tmp_MS4/report' % input_path )
  # except:
  #   print('Export report failed')

