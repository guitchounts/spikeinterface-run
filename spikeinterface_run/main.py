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

@cli.command(name="launch")
@click.option('--cores', type=int, default=8, help="Number of cores")
@click.option('--memory', type=str, default="32GB", help="RAM string")
@click.option('--wall-time', type=str, default='4:00:00', help="Wall time")
@click.option('--partition', type=str, default='short', help="Partition name")
@click.option('--sorter', type=str, default='ms4', help="Sorter name. Defult = ms4")
@click.option('--log-name', type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help='File name in which to store slurm logs.')
@click.option('--skip-sorted', type=bool, default=True, help='Bool to skip already sorted files. Default: True')

@shared_options
def launch(input_path, log_name, cores, memory, wall_time, partition, sorter, skip_sorted, sorter_path): #  # 
    
    print('Launching Sorting')
    

    # get all open ephys files in the input_path:
    open_ephys_folders = glob('%s/**/Record*' % input_path,recursive=True)
    
    if skip_sorted:
        # find sessions that already have the sorter_path in them and skip:
        open_ephys_folders = [folder for folder in open_ephys_folders if not os.path.exists('%s/../%s' % (folder,sorter_path) ) ]


    if len(open_ephys_folders) > 0:

        for session in open_ephys_folders:
            
            
            ### edit session name to place spaces with \ in the string (i.e. in ".../Record Node 101/...")
            session = '\\ '.join(session.split(' '))
            
            print(session)

            ## create log path:
            log_path = '%s/%s_%%j.out' % (session,log_name)
            

            os.system('sbatch -p {} -t {} --mem {} -c {} -o {} --wrap """spikeinterface-run submit {} --sorter-path {} """ '.format(
                            partition, wall_time, memory, cores, log_path, session, sorter_path
                      ))

      
            



    else:
        print('No sessions with ephys data found. Quitting...')


@cli.command(name="submit")
@shared_options
def submit(input_path,sorter_path):
    
    print('Submitting sorting on input path %s' % input_path)

    run_sorting(input_path,sorter_path)    






