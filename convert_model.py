import os
import numpy as np
import tensorflow as tf

import config_convert as config
import tfutil
import misc

# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def convert_to_npz(
    resume_run_id           = None,         # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,         # Snapshot index to resume training from, None = autodetect.
    *args,
    **kargs
    ):
    # Construct networks.
    with tf.device('/gpu:0'):
        network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
        print('Loading networks from "%s"...' % network_pkl)
        G, D, Gs = misc.load_pkl(network_pkl)
        G.save_npz('G.npz')
        D.save_npz('D.npz')
        Gs.save_npz('Gs.npz')


if __name__ == "__main__":
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    print('Running %s()...' % config.train['func'])
    tfutil.call_func_by_name(**config.train)
    print('Exiting...')

#----------------------------------------------------------------------------
