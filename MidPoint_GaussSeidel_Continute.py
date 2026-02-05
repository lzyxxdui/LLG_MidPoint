# -*- coding: utf-8 -*-
# %%
"""
Created on Wed Jul 26 16:04:00 2023

@author: zjjis
"""

# %%
import numpy as np
from scipy.sparse import csr_matrix
from LLG_3D import LLG_3D

# %%
class MidPoint_GaussSeidel_Continute(LLG_3D):
    def __init__(self, Parameter):
        LLG_3D.__init__(self, **Parameter)
        self.PrepareForUpperTriangle()

    def PrepareForUpperTriangle(self):
        Mark = np.arange(self.NC, dtype = int)
        self.I100 = np.concatenate((Mark+self.NC, Mark, Mark))
        self.J221 = np.concatenate((Mark+2*self.NC, Mark+2*self.NC, Mark+self.NC))
        return True

    def UpperMatrixLInv(self, m0, I):
        Value = np.zeros(shape = m0.shape, dtype = np.float64)
        Value[0: self.NC] = - self.alpha*m0[0: self.NC]
        Value[self.NC: 2*self.NC] = self.alpha*m0[self.NC: 2*self.NC] + self.alpha**2 * m0[0: self.NC] * m0[2*self.NC: ]
        Value[2*self.NC: ] = - self.alpha*m0[2*self.NC: ]
        E = csr_matrix((Value.flat,(self.I100, self.J221)), shape=(3*self.NC, 3*self.NC))
        return E + I

    def LowMatrixRMinus(self, m0):
        Value = self.alpha * m0
        Value[self.NC: 2*self.NC] = - Value[self.NC: 2*self.NC]
        E = csr_matrix((Value.flat,(self.J221, self.I100)), shape=(3*self.NC, 3*self.NC))
        return E
    
    def WhileInSolve(self, m0, Linm1, NumStep, I, Hm0half, HLinm1):

        # Prepare for the Stop Energy Tol # The energy of the last m!!  Note the choose the tn!!!
        if self.StopEnergyTol is not None: Energy0 = self.DiscreteEnergy(m0, tn = self.tn) 

        while True:
            # Judge if there exists Forcing Item
            # if self.ForcingItem is not None: 
            if self.ATExample is not None: 
                RForcing = self.Deltat * self.ForcingItem(self.CellCenterNode, self.tn + self.Deltat* (NumStep-1/2)).reshape(-1)
            
            # Solving m1
            InvU = self.UpperMatrixLInv(m0, I)
            LMinus = self.LowMatrixRMinus(m0)
            while True:
                m1m0D2 = (m0 + Linm1)/2
                Hm1m0D2 = (Hm0half + HLinm1)/2 
                R = m0 - self.Deltat * self.OutProduct(m1m0D2, Hm1m0D2) + LMinus@Linm1
                # if self.ForcingItem is not None: 
                if self.ATExample is not None: R += RForcing
                m1 = InvU@R
                if np.max(np.abs( Linm1 - m1 )) < self.tol: break
                print('err', np.max(np.abs( Linm1 - m1 )))
                Linm1 = m1.copy()
                HLinm1 = self.EffectiveField(Linm1, tn = self.tn + self.Deltat*(NumStep - 1/2))

            print('NumStep', NumStep)
            
            # Show the Error with a predetermined exact solution in each step
            if NumStep == self.MaxNumStep:
                if self.ATExample is not None:
                    Err = np.max(np.abs( self.ExactFun(self.CellCenterNode, tn = self.tn + NumStep*self.Deltat).reshape(-1) - m1 ))
                    print('Err', Err)
            
                 
            # Save Data and Figure, and show Figure
            if NumStep% self.StepFigurePer == 0:
                j = NumStep//self.StepFigurePer
                self.SaveDataAndFig(m = m1, j = j)
  
            ## Judge if Break
            NumStep += 1
            if NumStep > self.MaxNumStep: break
            if self.StopMTol is not None and self.JudgeIfBreakOnM(m0 = m0, m1 = m1) == False: break
            

            # Prepare next recursion
            m0 = m1.copy()
            Linm1 = m1.copy()
            Hm0half, HEachm0 = self.EffectiveField(m0, tn = self.tn + self.Deltat*(NumStep - 1/2), FullShow = True)
            HLinm1 = Hm0half.copy()

            ## Judge if Break
            if self.StopMxhTol is not None or self.StopEnergyTol is not None:
                if 3 in self.TypeEffective: 
                    HEachm0[self.TypeEffective.index(3), :] = self.ComputeHExternal(tn = self.tn + self.Deltat*(NumStep -1))
                    Hm1 = np.sum(HEachm0, axis = 0)
                else:
                    Hm1 = Hm0half.copy()
                if self.StopMxhTol is not None and self.JudgeIfBreakOnMxh(m1 = m1, Hm1 = Hm1) == False: break
                if self.StopEnergyTol is not None:
                    Energy1 = self.DiscreteEnergyWithH(m1, HEach = HEachm0)
                    if self.JudgeIfBreakOnEnergy(Energy0 = Energy0, Energy1 = Energy1) == False: break
                    Energy0 = Energy1
        self.m1, self.NumStep = m1.copy(), NumStep
        return m1

    def Solve(self, m0, tn = None, tol = 1e-12, MaxNumStep = None):
        # Reset temporal step size
        self.Deltat = self.Deltat * self.gamma * self.Ms
        self.tol = tol
        if tn is None: 
            self.tn = self.TimeDomain[0]
        else:
            self.tn = tn
        if MaxNumStep is not None: self.MaxNumStep = MaxNumStep
        
        # Save the Inital m and the according Figure (and show it)
        self.SaveDataAndFig(m = m0, j = 0)
        
        # Prepare for the Solving
        NumStep = 1
        I = self.Identity()
        Hm0half = self.EffectiveField(m0, tn = self.tn + self.Deltat*(NumStep - 1/2))
        m1 = self.WhileInSolve(m0, m0, NumStep, I, Hm0half, Hm0half)
        self.SaveDataAndFig(m1, "Last")
        return m1


    def Solve_Continute(self, tol = None, MultipleDeltat = None, MaxNumStep = None, StopMTol = None, StopEnergyTol = None, StopMxhTol = None):
        # Prepare for the Solving
        self.tn = self.tn + (self.NumStep - 1)*self.Deltat
        if tol is not None: self.tol = tol

        if MultipleDeltat is not None: self.Deltat = MultipleDeltat*self.Deltat
        if MaxNumStep is not None: self.MaxNumStep = MaxNumStep
        if StopMTol is not None: self.StopMTol = StopMTol
        if StopEnergyTol is not None: self.StopEnergyTol = StopEnergyTol
        if StopMxhTol is not None: self.StopMxhTol = StopMxhTol

        I = self.Identity()
        Hm0half = self.EffectiveField(self.m1, tn = self.tn + self.Deltat*(self.NumStep - 1/2))

        m1 = self.WhileInSolve(self.m1, self.m1, self.NumStep, I, Hm0half, Hm0half)
        self.SaveDataAndFig(m1, "Last")
        return m1


