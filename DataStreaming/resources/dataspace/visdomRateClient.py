from __future__ import print_function

import argparse
import numpy as np
import pickle
import os.path
from visdom import Visdom

# General parameters
DEFAULT_VISDOM_HOST = "127.0.0.1"
DEFAULT_VISDOM_PORT = 8097
DEFAULT_DATA_VALUE = 0.00

WIN_LINE = 'rate'
store_file = "rate.pckl"

parser = argparse.ArgumentParser(description='visdom client')
parser.add_argument('--visdom_host', type=str, default=DEFAULT_VISDOM_HOST, 
                    help='IP of the visdom server')
parser.add_argument('--visdom_port', type=int, default=DEFAULT_VISDOM_PORT, 
                    help='IP port of the visdom server')
parser.add_argument('--value', type=float, default=DEFAULT_DATA_VALUE, 
                    help='Y value for the line plot')
args = parser.parse_args()

print("Connecting to visdom server on ",args.visdom_host,":",args.visdom_port)
value = args.value

viz = Visdom(server="http://"+args.visdom_host, port=args.visdom_port)
assert viz.check_connection()

if not viz.win_exists(WIN_LINE):
  viz.line(Y = np.array([1]), X = np.array([1]),
                          opts = dict(
                                  xlabel = 'Iteration',
                                  ylabel = 'Rate',
                                  title = 'Bitcoin to Chinese Yuan',
                                  ), win=WIN_LINE
                          )


if os.path.exists(store_file):
  f = open(store_file, 'rb')
  iteration = int(pickle.load(f))
  f.close()
else:
  iteration = 0

print(value,iteration)
if iteration == 0:
  viz.line(Y = np.array([value]), X = np.array([iteration]), win = WIN_LINE, update='replace')
else:
  viz.line(Y = np.array([value]), X = np.array([iteration]), win = WIN_LINE, update='append')

iteration = iteration + 1
f = open(store_file, 'wb')
pickle.dump(iteration, f)
f.close()
