#Katie Swanson
#proj 2

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp

def main():
    hcX, hxY = hill_climb(z, 0.01, -2.5, 2.5, -2.5, 2.5)
    hcrrX, hcrrY = hill_climb_random_restart(z, 0.01, 1000, -2.5, 2.5, -2.5, 2.5)
    saX, saY = simulated_annealing(z, 0.01, 1.0, -2.5, 2.5, -2.5, 2.5)


def z(x, y):
    r = math.sqrt(x**2 + y**2)
    return (math.sin(x**2+3*y**2)/(0.1+r**2)) + (x**2 + 5*y**2)*(math.exp(1-r**2)/2)

def np_z(x, y):
    r = np.sqrt(x**2 + y**2)
    return (np.sin(x**2+3*y**2)/(0.1+r**2)) + (x**2 + 5*y**2)*(np.exp(1-r**2)/2)

def insert(arr, function_to_optimize, x, y, xmin, ymin, xmax, ymax):
    temp = arr
    if (x >= xmin and x <= xmax and y >= ymin and y <= ymax):
            temp.append([function_to_optimize(x, y), x, y])
    return temp

def hill_climb(function_to_optimize, step_size, xmin, xmax, ymin, ymax):
    f = []
    converged = False
    x = xmin
    y = ymin
    Xs = []
    Ys = []
    Zs = []
    fValue = 0

    fig2 = plt.figure()                                                                    
    ax2 = fig2.add_subplot(111, projection='3d') 

    while not converged:
        f = []
        f = insert(f, function_to_optimize, x - step_size, y - step_size, xmin, ymin, xmax, ymax)
        f = insert(f, function_to_optimize, x - step_size, y, xmin, ymin, xmax, ymax)
        f = insert(f, function_to_optimize, x - step_size, y + step_size, xmin, ymin, xmax, ymax)
        f = insert(f, function_to_optimize, x, y - step_size, xmin, ymin, xmax, ymax)
        f = insert(f, function_to_optimize, x, y + step_size, xmin, ymin, xmax, ymax)
        f = insert(f, function_to_optimize, x + step_size, y - step_size, xmin, ymin, xmax, ymax)
        f = insert(f, function_to_optimize, x + step_size, y, xmin, ymin, xmax, ymax)
        f = insert(f, function_to_optimize, x + step_size, y + step_size, xmin, ymin, xmax, ymax)

        fValue = function_to_optimize(x, y)


        #if the middle point is lowest, you have found local minima, so stop
        allFLessThan = True
        for i in f:
            if fValue > i[0]:
                allFLessThan = False

        if allFLessThan:
            converged = True
        else:
            zMin = 100000000000000000000
            for i in f:
                if i[0] < zMin:
                    #sets the x and y for the next while loop turn, repositions to min
                    zMin = i[0]
                    x = i[1]
                    y = i[2]
            ax2.scatter(x, y, zMin)

    for j in range(-25, 25, 3):
        for k in range(-25, 25, 3):
            Xs.append(j/10.0)
            Ys.append(k/10.0)
            Zs.append(function_to_optimize(j/10.0, k/10.0))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(Xs, Ys)
    Z = interp.griddata((Xs, Ys), Zs, (X, Y), method='linear')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    plt.show()

    plt.show()

    return x, y

def hill_climb_random_restart(function_to_optimize, step_size, num_restarts, xmin, xmax, ymin, ymax):
    g = []
    converged = False
    x = random.uniform(xmin, xmax)
    y = random.uniform(ymin, ymax)
    Xs = []
    Ys = []
    Zs = []
    fValue = 0
    count = 0

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')

    while not converged:
        g = []

        fValue = function_to_optimize(x, y)

        g = insert(g, function_to_optimize, x - step_size, y - step_size, xmin, ymin, xmax, ymax)
        g = insert(g, function_to_optimize, x - step_size, y, xmin, ymin, xmax, ymax)
        g = insert(g, function_to_optimize, x - step_size, y + step_size, xmin, ymin, xmax, ymax)
        g = insert(g, function_to_optimize, x, y - step_size, xmin, ymin, xmax, ymax)
        g = insert(g, function_to_optimize, x, y + step_size, xmin, ymin, xmax, ymax)
        g = insert(g, function_to_optimize, x + step_size, y - step_size, xmin, ymin, xmax, ymax)
        g = insert(g, function_to_optimize, x + step_size, y, xmin, ymin, xmax, ymax)
        g = insert(g, function_to_optimize, x + step_size, y + step_size, xmin, ymin, xmax, ymax)


        #if the middle point is lowest, you have found local minima, so stop
        allFLessThan = True
        for i in g:
            if fValue > i[0]:
                allFLessThan = False

        if allFLessThan:
            converged = True
        else:
            if count >= num_restarts:
                converged = True
            else:
                x = random.uniform(xmin, xmax)
                y = random.uniform(ymin, ymax)
            ax2.scatter(x, y, function_to_optimize(x, y))

        count += 1

    for j in range(-25, 25, 3):
        for k in range(-25, 25, 3):
            Xs.append(j/10.0)
            Ys.append(k/10.0)
            Zs.append(function_to_optimize(j/10.0, k/10.0))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(Xs, Ys)
    Z =interp.griddata((Xs, Ys), Zs, (X, Y), method='linear')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    plt.show()

    plt.show()

    return x, y
    

#reference: http://katrinaeg.com/simulated-annealing.html (partially)
def simulated_annealing(function_to_optimize, step_size, max_temp, xmin, xmax, ymin, ymax):
    x = random.uniform(xmin, xmax)
    y = random.uniform(ymin, ymax)
    z = function_to_optimize(x, y)
    T_min = 0.00001
    T = max_temp
    alpha = 0.9
    Xs = []
    Ys = []
    Zs = []
    Xs.append(x)
    Ys.append(y)
    Zs.append(z)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')

    while T > T_min:
        i = 0
        while i < 100:
            #old z use old x and old y, function_to_optimize is my cost function 
            z = function_to_optimize(x, y)

            #change which param every other turn
            if i % 2 == 0:
                x = random.uniform(xmin, xmax)
            else:
                y = random.uniform(ymin, ymax)

            #use the new x, old y, keep one param, change one param
            newZ = function_to_optimize(x, y)
            
            #acceptance probability that the algorithm will randomly accept a worse solution
            ap = acceptance_probability(z, newZ, T)
            if ap > random.random():
                #new solution and new cost in one
                newZ = z
                ax2.scatter(x, y, newZ)
            else:
                ax2.scatter(x, y, z)
            i += 1

        T = T*alpha

    for j in range(-25, 25, 5):
        for k in range(-25, 25, 5):
            Xs.append(j/10.0)
            Ys.append(k/10.0)
            Zs.append(function_to_optimize(j/10.0, k/10.0))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(Xs, Ys)
    Z =interp.griddata((Xs, Ys), Zs, (X, Y), method='linear')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    plt.show()

    plt.show()

    return x, y
        

#reference: http://katrinaeg.com/simulated-annealing.html
def acceptance_probability(old_cost, new_cost, T):
    return math.exp(-(math.fabs(old_cost - new_cost))/T)


main()
