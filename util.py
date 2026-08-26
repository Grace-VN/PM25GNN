import yaml
import sys
import os
import numpy as np


proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
conf_fp = os.path.join(proj_dir, 'config.yaml')
with open(conf_fp) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


import platform

# filepath resolution, in priority order:
#   1. KNOWAIR_FP / RESULTS_DIR env vars, if set - lets you point at a Kaggle
#      Dataset mount (e.g. /kaggle/input/knowair/KnowAir.npy) or any other
#      location without editing this tracked file.
#   2. config.yaml's filepath.<hostname> entry, for an exact platform.node()
#      match (the original behavior, e.g. a lab GPU server or a named laptop).
#   3. config.yaml's filepath.default entry, or if that's absent, KnowAir.npy
#      and results/ right next to this file - works out of the box after a
#      plain `git clone` on a machine/container/notebook not in (2), as long
#      as you've placed KnowAir.npy there yourself (see README's Dataset
#      section - it's intentionally not committed to the repo, it's ~300MB).
_default_file_dir = {
    'knowair_fp': os.path.join(proj_dir, 'KnowAir.npy'),
    'results_dir': os.path.join(proj_dir, 'results'),
}
nodename = platform.node()
if 'KNOWAIR_FP' in os.environ or 'RESULTS_DIR' in os.environ:
    _base = config['filepath'].get('default', _default_file_dir)
    file_dir = {
        'knowair_fp': os.environ.get('KNOWAIR_FP', _base['knowair_fp']),
        'results_dir': os.environ.get('RESULTS_DIR', _base['results_dir']),
    }
elif nodename in config['filepath']:
    file_dir = config['filepath'][nodename]
else:
    file_dir = config['filepath'].get('default', _default_file_dir)


def main():
    pass


if __name__ == '__main__':
    main()
