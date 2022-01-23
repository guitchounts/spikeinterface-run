import click
from glob import glob
import tqdm
import os
import numpy as np
from datetime import datetime
import functools
from subprocess import call

from spikeinterface_run.utils import run_sorting


orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init



@click.group()
@click.version_option()
def cli():
    pass


def shared_options(func):
    @click.argument('input-path', type=str, default='./') 
    @click.option('--sorter-path', type=str, default='tmp_MS4', help="Folder name for sorting output. Default: tmp_MS4")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def slurm_options(func):
    @click.option('--cores', type=int, default=8, help="Number of cores")
    @click.option('--memory', type=str, default="32GB", help="RAM string")
    @click.option('--wall-time', type=str, default='4:00:00', help="Wall time")
    @click.option('--partition', type=str, default='short', help="Partition name")
    @click.option('--log-name', type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help='File name in which to store slurm logs.')
    @click.option('--slurm', type=bool, default=True, help='Bool to send job to slurm (True) or to run interactively (False). Default: True')    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command(name="launch")
@click.option('--sorter', type=str, default='ms4', help="Sorter name. Defult = ms4")
@click.option('--skip-sorted', type=bool, default=True, help='Bool to skip already sorted files. Default: True')
@shared_options
@slurm_options
def launch(input_path, log_name, cores, memory, wall_time, partition, sorter, skip_sorted, sorter_path, slurm): #  # 
    
    print('Launching Sorting')
    

    # get all open ephys files in the input_path (these are the ones with Record 10X folders):
    open_ephys_folders = [os.path.split(item)[0] for item in glob('%s/**/Record*' % input_path,recursive=True)]


    if skip_sorted:
        # find sessions that already have the sorter_path in them and skip:
        open_ephys_folders = [folder for folder in open_ephys_folders if not os.path.exists('%s/../%s' % (folder,sorter_path) ) ]


    if len(open_ephys_folders) > 0:

        for session in open_ephys_folders:
            
            
            

            
            
            
            if not slurm:
                os.system('  """spikeinterface-run submit {} --sorter-path {} """ '.format(session, sorter_path
                            
                      ))
            else:

                ## create log path:
                log_path = '%s/%s_%%j.out' % (session,log_name)

                print('Submitting sorting on session %s' % session)

                os.system('sbatch -p {} -t {} --mem {} -c {} -o {} --wrap """spikeinterface-run submit {} --sorter-path {} """ '.format(
                            partition, wall_time, memory, cores, log_path, session, sorter_path
                      ))

      
            



    else:
        print('No sessions with ephys data found. Quitting...')


@cli.command(name="submit")
@shared_options
def submit(input_path,sorter_path):

    run_sorting(input_path,sorter_path)    






@cli.command(name="sort-concat")
@click.option('--date-to-sort', default=None, help="Date for which to concatenate and sort sessions. Format: YYYY-MM-DD Default: None")
@shared_options
@slurm_options
def sort_concat(input_path,date_to_sort, log_name, cores, memory, wall_time, partition, sorter_path, slurm):
    
    print('Starting sort-concat:')
        

    # get all open ephys files in the input_path (these are the ones with Record 10X folders):
    sessions = [os.path.split(item)[0] for item in glob('%s/**/Record*' % input_path,recursive=True)]

    dates = np.array([datetime.strptime(sesh.split('_')[1] , '%Y-%m-%d') for sesh in sessions])
    sessions = np.array(sessions)[np.argsort(dates)]
    dates = dates[np.argsort(dates)]

    if date_to_sort:
        sessions = [sessions[i] for i in range(len(dates)) if datetime.strptime(date_to_sort,'%Y-%m-%d') == dates[i]]
        dates = [dates[i] for i in range(len(dates)) if datetime.strptime(date_to_sort,'%Y-%m-%d') == dates[i]]

        session_sets = [sessions]
        
    else:
        session_sets = []
        for i,date in enumerate(np.unique(dates)):
            date_set = np.where(dates == date)[0]
            #date_sets.append(date_set)
            session_sets.append(sessions[date_set])
        print(date,session_sets)

    for session_set in session_sets:

        ## create log path:
        log_path = '%s/%s_%%j.out' % (session_set[0],log_name)

        session_set = '___'.join(session_set) # have to join to pass as an argument

        if not slurm:

                submit_string = 'spikeinterface-run submit {} --sorter-path {}'.format(session_set, sorter_path )

                os.system(submit_string)
                      
        else:

            
            print('Submitting sorting on session %s' % session)

            os.system('sbatch -p {} -t {} --mem {} -c {} -o {} --wrap """spikeinterface-run submit """{}""" --sorter-path {} """ '.format(
                        partition, wall_time, memory, cores, log_path, session_set, sorter_path
                      ))



    

