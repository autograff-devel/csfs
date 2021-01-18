''' Global command-line arguments shared by scripts'''

import sys, argparse
import module_paths
import csfs.config as config

cfg = config.cfg

args = argparse.ArgumentParser(description='''CSF analysis scripts command line arguments''')
# hanzi_test_params_refined
args.add_argument('-i', '--input', type=str, default='./fonts_default.json',
                 help='''Determines the input directory for ttf or svg parsing, 
                 if a json file is specified instead the directory will be parsed from the json file using the `input path` entry (this works only with fonts)''')
args.add_argument('-o', '--output', type=str, default='../../output/csfs',
                 help='''Determines output image location''')
args.add_argument('--shape_size', type=float, default=cfg.shape_size, #200., 
                  help='''Default glyph size (for sampling)''')
args.add_argument('--type', type=str, default='ttf', 
                 help='''Input type''')
args.add_argument('--flip_y', type=bool, default=False,
                  help='''Flips y axis''')

args.add_argument('--save_figures', type=bool, default=True, 
                  help='''If true saves figures''')
args.add_argument('--save_data', type=bool, default=True, 
                  help='''If true saves data''')
args.add_argument('--start_from', type=str, default='', 
                 help='''Start from a given font''')
args.add_argument('--max_count', type=int, default='1000000', 
                 help='''Maximum number of items''')

# Needed to fool VSCODE during interactive sessions
args.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
##########

def parse():
    print("Parsing arguments")
    args_cfg = args.parse_args()
    config.setup_cfg(args_cfg)
    config.cfg.loaded = True
    
    # If running from terminal always save figures
    if sys.stdin.isatty():
        print("Running from terminal, saving figures to:")
        print(config.cfg.output)
        config.cfg.save_figures = True


# %%
