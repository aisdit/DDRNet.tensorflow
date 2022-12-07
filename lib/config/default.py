from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from yacs.config import CfgNode as CN


_C = CN()
_C.OUTPUT_DIR = ''
_C.IMAGE_SIZE = [2048, 1024]  # width * height
_C.BASE_SIZE = 2048
_C.MODEL_FILE = ''




def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

