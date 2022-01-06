# spikeinterface-run
Code to run spikeinterface spike sorting sessions as batch jobs on slurm. Relies on spikeinterface version 0.90+.

Installation:

1. `git clone https://github.com/guitchounts/spikeinterface-run`
2. `cd spikeinterface-run`
3. `pip install -e .`

Usage:

1. `spikeinterface-run launch path/to/open/ephys/files/` that contain `Record Node 10X` directories.
