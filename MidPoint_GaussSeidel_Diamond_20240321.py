# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:03:23 2024

@author: zjjis
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:33:46 2024

@author: zjjis
"""

import numpy as np
# from MidPoint_GaussSeidel_Continute import MidPoint_GaussSeidel_Continute
from MidPoint import MidPoint
# In[]


SpaceDomain = [0, 200e-9, 0, 100e-9, 0, 20e-9] # 从原点0开始计算，长宽高尺寸
TimeDomain = [0, 1]


Nx = 100
Ny = 50
Nz = 1
t = 1e12 #把时间分成多少份


alpha = 0.1
Ms = 8.0e5
Cex = 1.3e-11
gamma = 2.211e5

TypeEffective = (0, 1, 2) # 选择使用哪一些场，0 - 

MaxNumStep = 10000

FigureName = None
FileName = None #"D:\OneDrive\Documents\LearningAndTeaching\Python\Micromagnetics_FDM\GroundState\Diamond\Data_MidPoint_GaussSeidel_Diamond_20240118.npz"
DataName = None #"D:\OneDrive\Documents\LearningAndTeaching\Python\Micromagnetics_FDM\GroundState\Diamond\m_MidPoint_GaussSeidel_Diamond_20240118"
StepFigurePer = 200
EnergyShow = True


Ku = 500 
AnisotropyDirection = np.array([1, 0, 0], dtype = np.float64)

StopMxhTol = 1e-4
StopMTol = None
StopEnergyTol = None

print('Nx', Nx, 'Ny', Ny, 'Nz', Nz, 't', t)


Par = {'SpaceDomain': SpaceDomain, 'TimeDomain': TimeDomain, 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 't': t, \
        'alpha': alpha, 'Ms': Ms, 'Cex': Cex, 'gamma': gamma, 'TypeEffective': TypeEffective,\
        "MaxNumStep" : MaxNumStep, "EndTime" : None, "StopEnergyTol" : StopEnergyTol, "StopEnergyType" : 0, "StopMTol" : StopMTol, "StopMxhTol": StopMxhTol,\
        "FileName" : FileName, "DataName" : DataName, "StepFigurePer" : StepFigurePer, \
        "Ku" : Ku, "AnisotropyDirection" : AnisotropyDirection, "He" : None}

    
Model = MidPoint(Parameter = Par)


# In[]

m0 = np.zeros((3, Nx, Ny, Nz), dtype = np.float64)
m0[0, 0:int((Nx)/2), 0:int((Ny)/2), :] = -1
m0[0, 0:int((Nx)/2), int((Ny)/2):, :] = 1
m0[0, int((Nx)/2):, 0:int((Ny)/2), :] = 1
m0[0, int((Nx)/2):, int((Ny)/2):, :] = -1


m0 = m0.reshape(-1)

# In[]

import time
StartTime = time.time()
m1 = Model.Solve(m0, tol = 1e-8)
EndTime = time.time()
print("Time", EndTime - StartTime) 


# In[]

print(Model.DiscreteEnergy(m1))
# Model.FigureForFace(m1)


# # In[]

# Model.MToOVF(m1)






