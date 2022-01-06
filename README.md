# spikeinterface-run
Code to run spikeinterface spike sorting sessions as batch jobs on slurm. Relies on spikeinterface version 0.90+.

Installation:

1. `git clone https://github.com/guitchounts/spikeinterface-run`
2. `cd spikeinterface-run`
3. `pip install -e .`

Usage:

1. `spikeinterface-run launch path/to/open/ephys/files/`. The path can be a an animal path that contains session folders (e.g. `/mouse/`) or a single session folder (e.g. `/mouse/session/`). The session folder should contain `Record Node 10X` Open Ephys directories. The script sends each session as a job to the slurm scheduler. The job runs the spike sorting, saves waveforms, and exports the data for curation in Phy.
