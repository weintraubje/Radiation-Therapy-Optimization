import numpy as np
import pandas as pd
import regex as re

def get_specs(folder_name):
    """Returns the specs of the data given a folder name"""
    
    f = open('task/task/' + folder_name + '/specs.txt')
    l = []
    
    for _ in range(0, 5):
        line = f.readline().replace('\n', '')
        print(line)
        l.append(int(re.sub("[^0-9]", "", line)))
    print('')
    
    return l

def load_data(folder_name, specs):
    """Given the specifications, load the critical, tumor, and beam maps"""
    
    # Load the critical and tumor map
    critical = np.loadtxt('task/task/' + folder_name + '/critical_raw.txt')
    print('Map Loaded: Critical')
    tumor = np.loadtxt('task/task/' + folder_name + '/tumor_raw.txt')
    print('Map Loaded: Tumor')
    
    # Load the beam map for each beam
    beams, first, last = [], 0, specs[1]
    for i in range(0, specs[0]):
        beam = np.loadtxt('task/task/' + folder_name + '/beam_raw.txt', 
                          skiprows = first,
                          max_rows = last)
        beams.append(beam)
        first = first + specs[1] + 1
    print('Map Loaded: Beams')
        
    return critical, tumor, beams