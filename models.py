import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from docplex.mp.model import Model


def build_model_1(specs, c, t, b):
    model = Model(name = 'm1')
    
    # Create ranges
    beam_range, vert_range, hori_range = range(0, specs[0]), range(0, specs[1]), range(0, specs[2])
    idx = [(i, j, k) for i in beam_range for j in vert_range for k in hori_range]
    
    # Add beam intensity variables and make non-negative
    x = model.continuous_var_list(keys=specs[0])
    for i in range(0, specs[0]):
        model.add_constraint(x[i] >= 0)
    print('Intensity variables added.')
        
    # Add beam intensity variables in (j,k) given beam (i)
    for j in vert_range:
        for k in hori_range:
            if c[j,k] == 1:
                if sum(b[i][j,k] for i in beam_range) == 0:
                    print('CRITICAL ERROR FOR', j, k)
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) <= specs[3])
            if t[j,k] == 1:
                if sum(b[i][j,k] for i in beam_range) == 0:
                    print('TUMOR ERROR FOR', j, k)
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) >= specs[4])
    print('Intensity constraints added.')
    
    # Objective Function
    obj = model.objective_expr
    for j in vert_range:
        for k in hori_range:
            obj += sum(x[i] * b[i][j,k] * (c[j,k] - t[j,k]) for i in beam_range)
    print('Object Function Constructed.')
    print(obj)
    
    model.minimize(obj)
    model.export_as_lp("test.lp")
    print('Model Exported.')
    
    model.print_information()
    solution = model.solve()
    if solution != None:
        print('Model Solved.')
    else:
        print('ERROR: NO SOLUTION')
    
    return solution



def build_model_2(specs, c, t, b):
    model = Model(name = 'm2', log_output=True)
    
    # Create ranges
    beam_range, vert_range, hori_range = range(0, specs[0]), range(0, specs[1]), range(0, specs[2])
    
    # Add beam intensity variables and make non-negative
    x = model.continuous_var_list(keys=specs[0])
    for i in range(0, specs[0]):
        model.add_constraint(x[i] >= 0)
    print('Intensity variables added.')
    
    # Add slack and surplus variables to allow for flexibility with critical and tumor regions
    c_x = model.continuous_var_list(keys=[('critical_slack', j, k) for j in vert_range for k in hori_range])
    for i in range(0, len(c_x)):
        model.add_constraint(c_x[i] >= 0)
        model.add_constraint(c_x[i] <= 2)
    s_x = model.continuous_var_list(keys=[('tumor_surplus', j, k) for j in vert_range for k in hori_range])
    for i in range(0, len(s_x)):
        model.add_constraint(s_x[i] >= 0)
        model.add_constraint(s_x[i] <= 10)

    # Add constraints given beam intensity variables, regions, and slack/surplus variables
    for j in vert_range:
        for k in hori_range:
            if c[j,k] == 1:
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) - c_x[j * specs[2] + k] <= specs[3])
            if t[j,k] == 1:
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) + s_x[j * specs[2] + k] >= specs[4])
    print('Intensity constraints added.')
        
    # Objective Function
    obj = model.objective_expr
    for j in vert_range:
        for k in hori_range:
            obj += sum(x[i] * b[i][j,k] * c[j,k] for i in beam_range)
    print('Object Function Constructed.')
    
    model.minimize(obj)
    model.export_as_lp("test.lp")
    
    print('Model Exported.')
    
    model.print_information()
    solution = model.solve()
    if solution != None:
        print('Model Solved.')
    else:
        print('ERROR: NO SOLUTION')
    
    return solution



def build_model_2_1(specs, c, t, b):
    model = Model(name = 'm1', log_output=True)
    
    # Create ranges
    beam_range, vert_range, hori_range = range(0, specs[0]), range(0, specs[1]), range(0, specs[2])
    
    # Add beam intensity variables and make non-negative
    x = model.continuous_var_list(keys=specs[0])
    for i in range(0, specs[0]):
        model.add_constraint(x[i] >= 0)
    print('Intensity variables added.')
    
    # Add slack and surplus variables to allow for flexibility with critical and tumor regions
    c_x = model.continuous_var_list(keys=[('critical_slack', j, k) for j in vert_range for k in hori_range])
    for i in range(0, len(c_x)):
        model.add_constraint(c_x[i] >= 0)
        model.add_constraint(c_x[i] <= 10)
    s_x = model.continuous_var_list(keys=[('tumor_surplus', j, k) for j in vert_range for k in hori_range])
    for i in range(0, len(s_x)):
        model.add_constraint(s_x[i] >= 0)
        model.add_constraint(s_x[i] <= 10)

    # Add constraints given beam intensity variables, regions, and slack/surplus variables
    for j in vert_range:
        for k in hori_range:
            if c[j,k] == 1:
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) - c_x[j * specs[2] + k] <= specs[3])
            if t[j,k] == 1:
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) + s_x[j * specs[2] + k] >= specs[4])
    print('Intensity constraints added.')
        
    # Objective Function
    obj = model.objective_expr
    for j in vert_range:
        for k in hori_range:
            obj += sum(x[i] * b[i][j,k] * c[j,k] for i in beam_range) \
                + c_x[j * specs[2] + k] * c[j,k] \
                + s_x[j * specs[2] + k] * t[j,k]
    print('Object Function Constructed.')
    
    model.minimize(obj)
    model.export_as_lp("test.lp")
    
    print('Model Exported.')
    
    model.print_information()
    solution = model.solve()
    if solution != None:
        print('Model Solved.')
    else:
        print('ERROR: NO SOLUTION')
    
    return solution



def build_model_3(specs, c, t, b, p_neighbor = 0.5):
    model = Model(name = 'm1', log_output=True)
    
    # Create ranges
    beam_range, vert_range, hori_range = range(0, specs[0]), range(0, specs[1]), range(0, specs[2])
    
    # Create critical neighbor map
    from scipy import ndimage
    c_neighbor = ndimage.generic_filter(c, np.nanmean, size = 3, mode='constant', cval=0)
    c_neighbor[c_neighbor > 0] = 1
    for j in vert_range:
        for k in hori_range:
            if t[j,k] == 0:
                c_neighbor[j,k] = c_neighbor[j,k] - c[j,k]
            
    # Add beam intensity variables and make non-negative
    x = model.continuous_var_list(keys=specs[0])
    for i in range(0, specs[0]):
        model.add_constraint(x[i] >= 0)
    print('Intensity variables added.')
    
    # Add slack and surplus variables to allow for flexibility with critical and tumor regions
    c_x = model.continuous_var_list(keys=[('critical_slack', j, k) for j in vert_range for k in hori_range])
    for i in range(0, len(c_x)):
        model.add_constraint(c_x[i] >= 0)
        model.add_constraint(c_x[i] <= 1)
    s_x = model.continuous_var_list(keys=[('tumor_surplus', j, k) for j in vert_range for k in hori_range])
    for i in range(0, len(s_x)):
        model.add_constraint(s_x[i] >= 0)
        model.add_constraint(s_x[i] <= 20)

    # Add constraints given beam intensity variables, regions, and slack/surplus variables
    for j in vert_range:
        for k in hori_range:
            if c[j,k] == 1:
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) - c_x[j * specs[2] + k] <= specs[3])
            if t[j,k] == 1:
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) + s_x[j * specs[2] + k] >= specs[4])
    print('Intensity constraints added.')
        
    # Objective Function
    obj = model.objective_expr
    for j in vert_range:
        for k in hori_range:
            obj += sum(x[i] * b[i][j,k] * (c[j,k] + p_neighbor * c_neighbor[j,k]) for i in beam_range) \
                + c_x[j * specs[2] + k] * c[j,k] \
                + s_x[j * specs[2] + k] * t[j,k]
    print('Object Function Constructed.')
    
    model.minimize(obj)
    model.export_as_lp("test.lp")
    
    print('Model Exported.')
    
    model.print_information()
    solution = model.solve()
    if solution != None:
        print('Model Solved.')
    else:
        print('ERROR: NO SOLUTION')
    
    return solution


def build_model_4(specs, c, t, b, p_neighbor = 0.5, p_regrow = 0.1):
    model = Model(name = 'm1', log_output=True)
    
    # Create ranges
    beam_range, vert_range, hori_range = range(0, specs[0]), range(0, specs[1]), range(0, specs[2])
    
    # Create critical neighbor map
    from scipy import ndimage
    c_neighbor = ndimage.generic_filter(c, np.nanmean, size = 3, mode='constant', cval=0)
    c_neighbor[c_neighbor > 0] = 1
    for j in vert_range:
        for k in hori_range:
            if t[j,k] == 0:
                c_neighbor[j,k] = c_neighbor[j,k] - c[j,k]
                
    # Slice out the interior of the tumor
    tr = ndimage.generic_filter(t, np.nansum, size = 10, mode='constant', cval=0)
    tr[tr != np.max(tr)] = 0
    tr[tr == np.max(tr)] = 1
    t = t - tr
            
    # Add beam intensity variables and make non-negative
    x = model.continuous_var_list(keys=specs[0])
    for i in range(0, specs[0]):
        model.add_constraint(x[i] >= 0)
    print('Intensity variables added.')
    
    # Add slack and surplus variables to allow for flexibility with critical and tumor regions
    c_x = model.continuous_var_list(keys=[('critical_slack', j, k) for j in vert_range for k in hori_range])
    for i in range(0, len(c_x)):
        model.add_constraint(c_x[i] >= 0)
        #model.add_constraint(c_x[i] <= 1)
    s_x = model.continuous_var_list(keys=[('tumor_surplus', j, k) for j in vert_range for k in hori_range])
    for i in range(0, len(s_x)):
        model.add_constraint(s_x[i] >= 0)
        model.add_constraint(s_x[i] <= 20)

    # Add constraints given beam intensity variables, regions, and slack/surplus variables
    for j in vert_range:
        for k in hori_range:
            if c[j,k] == 1:
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) - c_x[j * specs[2] + k] <= specs[3])
            if t[j,k] == 1:
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) + s_x[j * specs[2] + k] >= specs[4])
    print('Intensity constraints added.')
        
    # Objective Function
    obj = model.objective_expr
    for j in vert_range:
        for k in hori_range:
            obj += sum(x[i] * b[i][j,k] * (c[j,k] + p_neighbor * c_neighbor[j,k]) for i in beam_range) \
                + c_x[j * specs[2] + k] * c[j,k] \
                + s_x[j * specs[2] + k] * (t[j,k] + tr[j,k] * p_regrow)
    print('Object Function Constructed.')
    
    model.minimize(obj)
    model.export_as_lp("test.lp")
    
    print('Model Exported.')
    
    model.print_information()
    solution = model.solve()
    if solution != None:
        print('Model Solved.')
    else:
        print('ERROR: NO SOLUTION')
    
    return solution, t



def build_model_5(specs, c, t, b, p_neighbor = 0.5, p_regrow = 0.1, intensity = 0.75):
    model = Model(name = 'm1', log_output=True)
    
    # Create ranges
    beam_range, vert_range, hori_range = range(0, specs[0]), range(0, specs[1]), range(0, specs[2])
    
    # Create critical neighbor map
    from scipy import ndimage
    c_neighbor = ndimage.generic_filter(c, np.nanmean, size = 3, mode='constant', cval=0)
    c_neighbor[c_neighbor > 0] = 1
    for j in vert_range:
        for k in hori_range:
            if t[j,k] == 0:
                c_neighbor[j,k] = c_neighbor[j,k] - c[j,k]
                
    # Slice out the interior of the tumor
    tr = ndimage.generic_filter(t, np.nansum, size = 10, mode='constant', cval=0)
    tr[tr != np.max(tr)] = 0
    tr[tr == np.max(tr)] = 1
    t = t - tr
    
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
                
    # Add beam intensity variables and make non-negative
    x = model.continuous_var_list(keys=specs[0])
    for i in range(0, specs[0]):
        model.add_constraint(x[i] >= 0)
    x_l = model.continuous_var_list(keys=specs[0])
    for i in range(0, specs[0]):
        model.add_constraint(x_l[i] >= 0)
    x_r = model.continuous_var_list(keys=specs[0])
    for i in range(0, specs[0]):
        model.add_constraint(x_r[i] >= 0)
    print('Intensity variables added.')
    
    # Add slack and surplus variables to allow for flexibility with critical and tumor regions
    c_x = model.continuous_var_list(keys=[('critical_slack', j, k) for j in vert_range for k in hori_range])
    for i in range(0, len(c_x)):
        model.add_constraint(c_x[i] >= 0)
        model.add_constraint(c_x[i] <= 2)
    s_x = model.continuous_var_list(keys=[('tumor_surplus', j, k) for j in vert_range for k in hori_range])
    for i in range(0, len(s_x)):
        model.add_constraint(s_x[i] >= 0)
        model.add_constraint(s_x[i] <= 20)

    # Add constraints given beam intensity variables, regions, and slack/surplus variables
    for j in vert_range:
        for k in hori_range:
            if c[j,k] == 1:
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) \
                                   + model.sum(x_l[i] * left_b[i][j,k] for i in beam_range) \
                                   + model.sum(x_r[i] * right_b[i][j,k] for i in beam_range) \
                                   - c_x[j * specs[2] + k] <= specs[3])
            if t[j,k] == 1:
                model.add_constraint(model.sum(x[i] * b[i][j,k] for i in beam_range) \
                                   + model.sum(x_l[i] * left_b[i][j,k] for i in beam_range) \
                                   + model.sum(x_r[i] * right_b[i][j,k] for i in beam_range) \
                                     + s_x[j * specs[2] + k] >= specs[4])
    print('Intensity constraints added.')
        
    # Objective Function
    obj = model.objective_expr
    for j in vert_range:
        for k in hori_range:
            obj += sum(x[i] * b[i][j,k] * (c[j,k] + p_neighbor * c_neighbor[j,k]) for i in beam_range) \
                + sum(x_l[i] * left_b[i][j,k] * (c[j,k] + p_neighbor * c_neighbor[j,k]) for i in beam_range) \
                + sum(x_r[i] * right_b[i][j,k] * (c[j,k] + p_neighbor * c_neighbor[j,k]) for i in beam_range) \
                + c_x[j * specs[2] + k] * c[j,k] \
                + s_x[j * specs[2] + k] * (t[j,k] + tr[j,k] * p_regrow)
    print('Object Function Constructed.')
    
    model.minimize(obj)
    model.export_as_lp("test.lp")
    
    print('Model Exported.')
    
    model.print_information()
    solution = model.solve()
    if solution != None:
        print('Model Solved.')
    else:
        print('ERROR: NO SOLUTION')
    
    return solution, t 