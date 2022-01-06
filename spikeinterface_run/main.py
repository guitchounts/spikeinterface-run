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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@cli.command(name="launch")
@click.option('--slurm/--no-slurm', default=False, help='Submit a slurm job for each .mkv file')
@click.option('--cores', type=int, default=4, help="Number of cores")
@click.option('--memory', type=str, default="8GB", help="RAM string")
@click.option('--wall-time', type=str, default='30:00', help="Wall time")
@click.option('--partition', type=str, default='short', help="Partition name")
@click.option('--sorter', type=str, default='ms4', help="Sorter name. Defult = ms4")
@click.option('--sorting_path', type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help='Path to store slurm logs.')
@shared_options
def launch(input_path, sorting_path, slurm, cores, memory, wall_time, partition, sorter): #  # 
    
    print('Launching Sorting')
    

    # get all open ephys files in the input_path:
    open_ephys_folders = glob('%s/**/Record*' % input_path,recursive=True)
    
    

    if len(open_ephys_folders) > 0:

        for session in open_ephys_folders:
            
            
            ### edit session name to place spaces with \ in the string (i.e. in ".../Record Node 101/...")
            session = '\\ '.join(session.split(' '))
            
            print(session)
            

            os.system('sbatch -p {} -t {} --mem {} -c {} --wrap """spikeinterface-run submit {}""" '.format(
                            partition, wall_time, memory, cores, session
                      ))

      
            



    else:
        print('No sessions with ephys data found. Quitting...')


@cli.command(name="submit")
@shared_options
def submit(input_path):
    
    print('Submitting sorting on input path %s' % input_path)

    run_sorting(input_path)    






