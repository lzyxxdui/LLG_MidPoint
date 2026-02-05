# -*- coding: utf-8 -*-
# %%
"""
Created on Wed Jul 26 16:04:00 2023

@author: zjjis
"""
'''
Wrong!!!!
# %%
import numpy as np
from scipy.sparse.linalg import spsolve
from LLG_3D import LLG_3D
import pyamg

# %%
class MidPoint_AMG(LLG_3D):
    def __init__(self, Parameter):
        LLG_3D.__init__(self, **Parameter)
    
    def Solve(self, m0, tn = None, tol = 1e-12, MaxNumStep = None, TimeScale = True):
        # Reset temporal step size
        if TimeScale == True: self.Deltat = self.Deltat * self.gamma * self.Ms
        
        # Save the Inital m and the according Figure (and show it)
        self.SaveDataAndFig(m = m0, j = 0)

        # Mark the beginning time
        if tn is None: tn = self.TimeDomain[0]
        if MaxNumStep is None: MaxNumStep = self.MaxNumStep
                
        # Prepare for the Stop Energy Tol # The energy of the last m!!  Note the choose the tn!!!
        if self.StopEnergyTol is not None: Energy0 = self.DiscreteEnergy(m0, tn = tn) 
            
        # Prepare for the Solving
        NumStep = 1
        Linm1 = m0.copy()
        I = self.Identity()
        Hm0half = self.EffectiveField(m0, tn = tn + self.Deltat*(NumStep - 1/2))
        HLinm1 = Hm0half.copy()
        while True:
            # Judge if there exists Forcing Item
            # if self.ForcingItem is not None: 
            if self.ATExample is not None: 
                RForcing = self.Deltat * self.ForcingItem(self.CellCenterNode, tn + self.Deltat* (NumStep-1/2)).reshape(-1)
            
            # Solving m1
            L = I - self.alpha * self.FunMatrixofmnTimes(m0)
            print('L', L.shape)
            AMG_Solver = pyamg.ruge_stuben_solver(L, max_coarse=10, max_levels=5, keep=True)
                
            while True:
                m1m0D2 = (m0 + Linm1)/2
                Hm1m0D2 = (Hm0half + HLinm1)/2     
                R = m0 - self.Deltat * self.OutProduct(m1m0D2, Hm1m0D2)
                # if self.ForcingItem is not None: 
                if self.ATExample is not None: R += RForcing
                # m1 = spsolve(L, R)
                AMG_Solver.solve(R, tol = 1e-10)
                if np.max(np.abs( Linm1 - m1 )) < tol: break
                Linm1 = m1.copy()
                HLinm1 = self.EffectiveField(Linm1, tn = tn + self.Deltat*(NumStep - 1/2))

            print('NumStep', NumStep)
      
            # Show the Error with a predetermined exact solution in each step
            # if NumStep == MaxNumStep:
            #     if self.ATExample is not None:
            #         Err = np.max(np.abs( self.ExactFun(self.CellCenterNode, tn = tn + NumStep*self.Deltat).reshape(-1) - m1 ))
            #         print('Err', Err)
            if self.ATExample is not None:
                Err = np.max(np.abs( self.ExactFun(self.CellCenterNode, tn = tn + NumStep*self.Deltat).reshape(-1) - m1 ))
                print('Err', Err)
                
            # Save Data and Figure, and show Figure
            if NumStep% self.StepFigurePer == 0:
                j = NumStep//self.StepFigurePer
                self.SaveDataAndFig(m = m1, j = j)
            
            ## Judge if Break
            if self.StopMTol is not None and self.JudgeIfBreakOnM(m0 = m0, m1 = m1) == False: break
            NumStep += 1
            if NumStep > MaxNumStep: break

            # Prepare next recursion
            m0 = m1.copy()
            Linm1 = m1.copy()
            Hm0half, HEachm0 = self.EffectiveField(m0, tn = tn + self.Deltat*(NumStep - 1/2), FullShow = True)
            HLinm1 = Hm0half.copy()

            ## Judge if Break
            if self.StopMxhTol is not None or self.StopEnergyTol is not None:
                if 3 in self.TypeEffective: 
                    HEachm0[self.TypeEffective.index(3), :] = self.ComputeHExternal(tn = tn + self.Deltat*(NumStep -1))
                    Hm1 = np.sum(HEachm0, axis = 0)
                else:
                    Hm1 = Hm0half.copy()
                if self.StopMxhTol is not None and self.JudgeIfBreakOnMxh(m1 = m1, Hm1 = Hm1) == False: break
                if self.StopEnergyTol is not None:
                    Energy1 = self.DiscreteEnergyWithH(m1, HEach = HEachm0)
                    if self.JudgeIfBreakOnEnergy(Energy0 = Energy0, Energy1 = Energy1) == False: break
                    Energy0 = Energy1
    
        self.SaveDataAndFig(m1, "Last")
        return m1

'''