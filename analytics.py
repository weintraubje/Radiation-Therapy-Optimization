import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def calc_m(sol, b, print_vars = False):
    """Calculate the sum of radiation in each cell of the matrix"""

    sol_l = []
    for var, value in sol.iter_var_values():
        if str(var)[0] == 'x':
            if print_vars == True:
                print(var, value)
            sol_l.append([str(var)[1:], value])

    first = True
    for i in range(0, len(sol_l)):
        if first == True:
            first = False
            m = b[int(sol_l[i][0])-1].copy() * sol_l[i][1]
        else:
            m = m + b[int(sol_l[i][0])-1].copy() * sol_l[i][1]    
    
    return m



def plot_beams(sol, b, c, t, cmap_choice = 'magma', print_vars = False, magnetic = False):
    """Plot the path of the beams in python"""

    # Get basic matrices
    if magnetic == False:
        m = calc_m(sol, b, print_vars)
    else:
        m = magnetic_calc_m(sol, b, print_vars)
    m_c_empty, m_t_empty = False, False

    from matplotlib.colors import colorConverter
    import matplotlib as mpl
    color1 = colorConverter.to_rgba('green')
    color2 = colorConverter.to_rgba('red')
    critical_cmap = mpl.colors.LinearSegmentedColormap.from_list('critical_cmap',[color1,color2],256)
    critical_cmap._init()
    alpha_c = np.linspace(0, 0.25, critical_cmap.N+3)
    critical_cmap._lut[:,-1] = alpha_c

    tumor_cmap = mpl.colors.LinearSegmentedColormap.from_list('critical_cmap',[color2,color1],256)
    tumor_cmap._init()
    alpha_t = np.linspace(0, 0.25, tumor_cmap.N+3)
    tumor_cmap._lut[:,-1] = alpha_t


    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(m, cmap=cmap_choice)
    ax.imshow(c, cmap=critical_cmap)
    ax.imshow(t, cmap=tumor_cmap)
    plt.axis('off')
    plt.title('Map of Radiation')
    plt.tight_layout()
    
    return None



def report_effectiveness(sol, b, c, t, max_rad = 2, min_rad = 10, print_vars = False, plot = True, magnetic = False):
    """See how good the particular model is at providing radiation coverage"""
    
    # Get basic matrices
    if magnetic == False:
        m = calc_m(sol, b, print_vars)
    else:
        m = magnetic_calc_m(sol, b, print_vars)
    m_c_empty, m_t_empty = False, False
    
    
    # Critical area
    m_c = [x for x in (m * c).flatten() if x > 0]
    sum_m_c = sum(m_c)
    if sum_m_c == 0:
        m_c_empty = True
    else:
        len_m_c, len_c = len(m_c), len([x for x in c.flatten() if x == 1])
        avg_m_c = sum_m_c / len_m_c
        acceptable_m_c = len([x for x in m_c if x <= max_rad]) - len_m_c + len_c
    
    
    # Tumor area
    m_t = [x for x in (m * t).flatten() if x > 0]
    sum_m_t = sum(m_t)
    if sum_m_t == 0:
        m_t_empty = True
    if sum_m_t == None:
        m_t_empty = True
    else:
        len_m_t, len_t = len(m_t), len([x for x in t.flatten() if x == 1])
        avg_m_t = sum_m_t / len_m_t
        acceptable_m_t = len([x for x in m_t if x >= min_rad]) - len_m_t + len_t
        
        
    # Print report
    print('\nMODEL REPORT\n')    
    if m_c_empty == False:
        print(str(round(sum_m_c,1))+' units of radiation were delivered to critical cells.')
        print(str(round(avg_m_c,1))+' units were delivered to each cell, on average.')
        print(str(acceptable_m_c)+' cells were found to have acceptable levels of radiation, out of '+str(len_c)+'.')
        print(str(round(100*acceptable_m_c / len_c,2))+'% of cells were found to have acceptable levels of radiation.\n')
    else:
        print('No units of radiation were delivered to any critical cells.\n')
        
    if m_t_empty == False:
        print(str(round(sum_m_t,1))+' units of radiation were delivered to tumor cells.')
        print(str(round(avg_m_t,1))+' units were delivered to each cell, on average.')
        print(str(acceptable_m_t)+' cells were found to have acceptable levels of radiation, out of '+str(len_t)+'.')
        print(str(round(100*acceptable_m_t / len_t,2))+'% of cells were found to have acceptable levels of radiation.')
    else:
        print('No units of radiation were delivered to any tumor cells.')
    
    
    # Plot if requested
    if plot == True:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(m*t, cmap='Greens')
        plt.axis('off')
        plt.title('Map of Tumor Delivery')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(m*c, cmap='Reds')
        plt.axis('off')
        plt.title('Map of Critical Delivery')
        plt.tight_layout()
        plt.show()
    
    return None



def magnetic_calc_m(sol, b, print_vars = False):
    """Calculate the sum of radiation in each cell of the matrix"""

    # Add two magnetic fields
    left_b, right_b = np.copy(b), np.copy(b)
    for beam in range(0, len(left_b)):
        intensity = 0.75
        for _ in range(0, len(left_b[beam])):
            if _ < len(left_b)/2:
                left_b[beam][_] = np.roll(left_b[beam][_],-int(_**intensity))
            else:
                left_b[beam][_] = np.roll(left_b[beam][_],-int((len(left_b[beam])-_)**intensity))
    for beam in range(0, len(right_b)):
        intensity = 0.75
        for _ in range(0, len(right_b[beam])):
            if _ < len(right_b)/2:
                right_b[beam][_] = np.roll(right_b[beam][_],int(_**intensity))
            else:
                right_b[beam][_] = np.roll(right_b[beam][_],int((len(right_b[beam])-_)**intensity))
    
    # All together
    b = np.concatenate((b, left_b, right_b), axis=0)
    
    
    sol_l = []
    for var, value in sol.iter_var_values():
        if str(var)[0] == 'x':
            if print_vars == True:
                print(var, value)
            sol_l.append([str(var)[1:], value])

    first = True
    for i in range(0, len(sol_l)):
        if first == True:
            first = False
            m = b[int(sol_l[i][0])-1].copy() * sol_l[i][1]
        else:
            m = m + b[int(sol_l[i][0])-1].copy() * sol_l[i][1]    
    
    return m



def plot_magnetic_shifts(a_b):
    """Plot the basic magnetic shifts."""

    data_1 = np.copy(a_b)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(sum(data_1), cmap='Blues')
    plt.axis('off')
    plt.title('No Shift')
    plt.tight_layout()
    plt.show()

    data_2 = np.copy(a_b)

    for beam in range(0, len(data_2)):
        intensity = 0.75
        for _ in range(0, len(data_2[beam])):
            if _ < len(data_2)/2:
                data_2[beam][_] = np.roll(data_2[beam][_],int(_**intensity))
            else:
                data_2[beam][_] = np.roll(data_2[beam][_],int((len(data_2[beam])-_)**intensity))

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(sum(data_2), cmap='Blues')
    plt.axis('off')
    plt.title('Shift Right: Magnet Intensity = '+str(intensity))
    plt.tight_layout()
    plt.show()

    data_3 = np.copy(a_b)

    for beam in range(0, len(data_3)):
        intensity = 0.75
        for _ in range(0, len(data_3[beam])):
            if _ < len(data_3)/2:
                data_3[beam][_] = np.roll(data_3[beam][_],-int(_**intensity))
            else:
                data_3[beam][_] = np.roll(data_3[beam][_],-int((len(data_3[beam])-_)**intensity))

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(sum(data_3), cmap='Blues')
    plt.axis('off')
    plt.title('Shift Left: Magnet Intensity = '+str(intensity))
    plt.tight_layout()
    plt.show()

    return None