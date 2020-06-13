from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Iterable
import argparse

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--method", help="Name of optimizer method (e.g., NestedLookahead Adam)", type=str, nargs='+', default="")
parser.add_argument("--with_t", help="Specify keywords of filenames to include (e.g., 200 epochs)", type=str, nargs='+', default="")
parser.add_argument("--without", help="Specify keywords of filenames not to include (e.g., pullback reset)", nargs='+', type=str, default="pullback reset")
parser.add_argument("--tag", help="Tag added to image file name (e.g., demo)", type=str, default="demo")
parser.add_argument("--root", help="directory name containing data to graph (default: accs)", type=str, default="accs")
parser.add_argument("--runs", help="Number of runs within data (default: 2)", type=int, default=2)
parser.add_argument("--img", help="Image file base name (default: test_acc)", type=str, default="test_acc")
args = parser.parse_args()

# Settings
tag = args.tag
root = Path(args.root)
runs = args.runs
img_name = args.img

name = 'CIFAR-10'
to_file_base = 'cifar10_' + tag
verbose = True

method = args.method # terms for data to include in plot
without_terms = args.without # terms for data not to include in plot (e.g., momentum=0, pullback, Nested, etc.)
with_terms = args.with_t # terms for data to include

# Ensure argparse accepts terms correctly
if isinstance(without_terms, str):
    without_terms = without_terms.split()
    
if isinstance(with_terms, str):
    with_terms = with_terms.split()
    
if not isinstance(method, str) and isinstance(method, Iterable):
    method = " ".join(method)

# Ensure `plots/` directory exists
if not os.path.isdir('plots'):
    os.mkdir('plots')

# Definitions
def queries(values, keys=None, without=None):
    "Get `values` that contain all `keys` without any values specified in `without`"
    if isinstance(keys, str):
        values = [v for v in values if keys in v]
    elif isinstance(keys, Iterable):
        for k in keys:
            values = [v for v in values if k in v]
    if isinstance(without, str):
        values = [v for v in values if without not in v]
    elif isinstance(without, Iterable):
        for n in without:
            values = [v for v in values if n not in v]
    return sorted(values)

def length_test(A):
    "Checks that all objects in A are the same length"
    length = len(A[0])
    for obj in A[1:]:
        try:
            assert length == len(obj)
        except AssertionError:
            raise ValueError(f"{obj} should be size {length} but was size {len(obj)}")
    if verbose: print("Length Test Passed\n")

def name_change(a):
    for i in range(len(a)):
        a[i] = a[i].replace('accs_','')
        a[i] = a[i].replace('_',' ')
        a[i] = a[i].replace(' a03',' (a=0.3,')
        a[i] = a[i].replace(' a07',' (a=0.7,')
        a[i] = a[i].replace('h07','h=0.7)')
        a[i] = a[i].replace('h03','h=0.3)')
        a[i] = a[i].replace(' k5',' (k=5,')
        a[i] = a[i].replace('s10','s=10)')
        a[i] = a[i].replace('s5','s=5)')
        a[i] = a[i].replace('k10','(k=10,')
        a[i] = a[i].replace(' half',' (half')
        a[i] = a[i].replace(' 5x',' (5x')
        a[i] = a[i].replace('mom0 wd0','(momentum=0, weight decay=0)')
        a[i] = a[i].replace(' mom0',' (momentum=0')
        a[i] = a[i].replace('lr','learning rate)')
        a[i] = a[i].replace('wd','weight decay=')
        a[i] = a[i].replace(' weight decay=0',' (weight decay=0)') if 'momentum' not in a[i] else a[i]
    return a

# Get file names
files = os.listdir(root)
test_files = queries(files, keys=['.csv','test'] + with_terms, without=without_terms)

if verbose: print(test_files, '\n')

experiments = sorted(list(set([i[5:-6] for i in test_files])))

# Separate file names by experiment
exp_test = [] # test_files (names) separated by experiment
for exp in experiments:
    if ('pullback' not in exp) and ('reset' not in exp):
        without=['pullback','reset']
    else:
        without=None
    
    exp_test.append(queries(test_files, keys=exp, without=without))
    
length_test(exp_test)

# Change experiment names for legends
experiments = name_change(experiments)
# filename_dict = dict(zip(sorted(test_files), sorted(experiments*2)))

if verbose: print(experiments, '\n')

# Load file data for each experiment
tests = [] # exp_test file data
for i in range(len(experiments)):
    test = []
    for p in exp_test[i]:
        p = root/Path(p)
        test.append(pd.read_csv(p, index_col=0).to_numpy())
    
    test = np.array(test)[:,:,0].T
    tests.append(test)
    
length_test(tests)
assert len(tests) == len(experiments)

# Create dictionary for legends --> data
experiment_dict = dict(zip(experiments, tests))

# Matplotlib colors
colorlist = ['sienna','darkgreen','royalblue','red',
             'orange','darkcyan','darkviolet','lime','gold','fuchsia','olive','darkred']

def get_style(exp):
    "Gets linestyle for given experiment"
    if 'pullback' in exp:
        s = '--'
    elif 'reset' in exp:
        s = '-.'
    else:
        s = '-'
    return s

def create_plot(method, data):
    """Generates and saves a plot
    
    method: optimizer name used in title and saved file
    experiment_type: legend names for each set of values"""
    title_font = {'size':'16'}
    axis_font = {'size':'14'}

    plt.figure(figsize=(15,10))
    plt.tick_params(labelsize=14)
    plt.title(f"{name} {method} Test Accuracies", **title_font)

    for i, exp in enumerate(experiments):
        styles = {'linestyle': get_style(exp), 'color': colorlist[i]}
        for i in range(runs):
            vals = experiment_dict[exp]
            plt.plot(vals[:,i], alpha=0.2, **styles)
        plt.plot(vals.mean(axis=1), label=exp, **styles)
        plt.xlabel('Steps', labelpad=10, **axis_font)
        plt.ylabel('Accuracy', labelpad=10, **axis_font)
        plt.legend(loc='lower right', prop=axis_font)

    plt.savefig(f"plots/{to_file_base}_{method.lower().replace(' ','_')}_{img_name}.png")
    print(f"saved to plots/{to_file_base}_{method.lower().replace(' ','_')}_{img_name}.png", '\n')
    plt.close()

data = queries(experiments, keys=method, without=without_terms)

create_plot(method, data)
