from arguments import args
import torch
import numpy as np
import random
import threading
import time
import os
import sys

import ipdb
st = ipdb.set_trace

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def main():
    print("Mode:", args.mode)
    print(type(args.mode))    
    if 'convnet_nsd_response_optimized' in args.mode:
        from run.convnet_fit_nsd import NSD_OPT
        nsd_optimize = NSD_OPT()
        nsd_optimize.run_train()
    elif 'convnet_nsd_eval_dissect' in args.mode:
        from run.convnet_eval_dissect_nsd import Eval
        eval_vit = Eval()
        eval_vit.run_dissection()
    elif 'convnet_places_eval_dissect' in args.mode:
        from run.convnet_eval_dissect_nsd import Eval
        eval_vit = Eval()
        eval_vit.run_dissection()
    elif 'convnet_gqa_eval_dissect' in args.mode:
        from run.convnet_eval_dissect_gqa import Eval_VG
        eval_vit = Eval_VG()
        eval_vit.run_dissection()
    elif 'convnet_xtc_eval_dissect' in args.mode:
        from run.convnet_eval_dissect_xtc import Eval_XTC
        eval_vit = Eval_XTC()
        eval_vit.run_dissection()
    else:
        raise NotImplementedError

    print("main finished.")
    
if __name__ == '__main__':
    main()