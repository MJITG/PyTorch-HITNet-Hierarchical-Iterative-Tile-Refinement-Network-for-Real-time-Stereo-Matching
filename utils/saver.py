import os
import shutil
import torch
from collections import OrderedDict
import glob
import torch.distributed as dist
import json


class Saver(object):

    def __init__(self, args, use_dist=False):
        self.args = args
        self.use_dist = use_dist
        # self.directory = os.path.join('run', args.dataset, args.checkname)
        self.directory = args.logdir
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))  # Sort saved results folders
        self.run_id = max([int(x.split('_')[-1]) for x in self.runs]) + 1 if self.runs else 0  # Create a new folder id
        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(self.run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        print('Saver currently runs in {}'.format(self.experiment_dir))

    def save_experiment_config(self):
        """Write experiment config to file"""

        if (self.use_dist and dist.get_rank() == 0) or not self.use_dist:
            logfile = os.path.join(self.experiment_dir, 'parameters.txt')
            log_file = open(logfile, 'w')
            log_file.write('\n')
            json.dump(self.args.__dict__, log_file, indent=2)
            log_file.write('\n')
            log_file.close()
