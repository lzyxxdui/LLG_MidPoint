import numpy as np
from scipy.sparse import csr_matrix
#from LLG_3D_StrayField import StrayField
from LLG_3D_StrayField_gpu import StrayField
from LLG_3D_FigureAndSaveData import FigureAndSaveData
from LLG_3D_DiscreteOperator import DiscreteOperator
from LLG_3D_AccuracyTestExample import AccuracyTestExample
# from scipy.sparse.linalg import spsolve
# import sys
# import os


def PythonToOOMMF(m, Nx, Ny, Nz, FileName = None, fmt = '%.011f'):
    NC = Nx*Ny*Nz
    Value = m.reshape(3, Nx, Ny, Nz)
    Value = np.transpose(Value, (3, 2, 1, 0)).reshape(NC, 3)
    if FileName is not None: np.savetxt(FileName, Value, fmt = fmt)
    return Value

# def MToOVF(m, Nx, Ny, Nz, Deltax, Deltay, Deltaz, Decimals = 11):
#     NC = Nx*Ny*Nz
#     MagOfM = np.linalg.norm(m.reshape(3, Nx, Ny, Nz), axis = 0)
#     Linm = (m.reshape(3, Nx, Ny, Nz)/MagOfM).round(decimals = Decimals)
#     Linm = np.transpose(Linm, (3, 2, 1, 0)).reshape(NC, 3)
#     with open("M_Field.ovf", "w") as file:
#         file.write("# OOMMF: rectangular mesh v1.0\n")
#         file.write("# Segment count: 1\n")
#         file.write("# Begin: Segment\n")
#         file.write("# Begin: Header\n")
#         file.write("# Title: M-Field\n")
#         file.write("# meshtype: rectangular\n")
#         file.write("# meshunit: m\n")
#         file.write(f"# xstepsize: {Deltax}\n")
#         file.write(f"# ystepsize: {Deltay}\n")
#         file.write(f"# zstepsize: {Deltaz}\n")
#         file.write(f"# xnodes: {Nx}\n")
#         file.write(f"# ynodes: {Ny}\n")
#         file.write(f"# znodes: {Nz}\n")
#         file.write("# End: Header\n")
#         file.write("# Begin: Data Text\n")
#         for i in range(NC): file.write(f"{Linm[i][0]} {Linm[i][1]} {Linm[i][2]}\n")
#         file.write("# End: Data Text\n")
#         file.write("# End: Segment\n")
#     return True


def MToOVF(m, Nx, Ny, Nz, Deltax, Deltay, Deltaz, Decimals=11):
    NC = Nx * Ny * Nz
    MagOfM = np.linalg.norm(m.reshape(3, Nx, Ny, Nz), axis=0)
    Linm = (m.reshape(3, Nx, Ny, Nz) / MagOfM).round(decimals=Decimals)
    Linm = np.transpose(Linm, (3, 2, 1, 0)).reshape(NC, 3)
    Path = "D:/OneDrive/Documents/LearningAndTeaching/Python/Micromagnetics_FDM/"
    with open(Path + "M_Field.ovf", "wb") as file:
        header = (
            "# OOMMF: rectangular mesh v1.0\n"
            "# Segment count: 1\n"
            "# Begin: Segment\n"
            "# Begin: Header\n"
            "# Title: M-Field\n"
            "# meshtype: rectangular\n"
            "# meshunit: m\n"
            f"# xstepsize: {Deltax}\n"
            f"# ystepsize: {Deltay}\n"
            f"# zstepsize: {Deltaz}\n"
            f"# xnodes: {Nx}\n"
            f"# ynodes: {Ny}\n"
            f"# znodes: {Nz}\n"
            "# End: Header\n"
            "# Begin: Data Text\n"
        )
        file.write(header.encode('utf-8'))
        
        for i in range(NC):
            line = f"{Linm[i][0]} {Linm[i][1]} {Linm[i][2]}\n"
            file.write(line.encode('utf-8'))
        
        footer = (
            "# End: Data Text\n"
            "# End: Segment\n"
        )
        file.write(footer.encode('utf-8'))
    
    return True



def OutProduct(A, B): # A = (3, N)
    LinA, LinB = A.reshape(3, -1), B.reshape(3, -1)
    C = np.zeros(LinA.shape, dtype = np.float64)
    C[0, :] = LinA[1, :] * LinB[2, :] - LinA[2, :] * LinB[1, :]
    C[1, :] = LinA[2, :] * LinB[0, :] - LinA[0, :] * LinB[2, :]
    C[2, :] = LinA[0, :] * LinB[1, :] - LinA[1, :] * LinB[0, :]
    return C.reshape(-1)


class LLG_3D(StrayField, FigureAndSaveData, DiscreteOperator, AccuracyTestExample):
    def __init__(self, SpaceDomain, TimeDomain, Nx, Ny, Nz, t, alpha, Ms, Cex, gamma, TypeEffective, \
                 MaxNumStep = None, EndTime = None, StopEnergyTol = None, StopEnergyType = 0, StopMTol = None, StopMxhTol = None, \
                 FileName = None, DataName = None, StepFigurePer = np.inf, TotalFigurePer = None, \
                 Ku = None, AnisotropyDirection = None, He = None, D = None, mu0 = 4 * np.pi * 1e-7, \
                 ATExample = None, ATExampleType = "LLG", IfSpaceScale = True, BCSetting = "niemann"):
        #######################
        self.SpaceDomain = SpaceDomain
        self.TimeDomain = TimeDomain
        self.Nx, self.Ny, self.Nz, self.t = Nx, Ny, Nz, t
        self.alpha, self.Ms, self.Cex, self.gamma, self.TypeEffective = alpha, Ms, Cex, gamma, TypeEffective

        #######################
        if MaxNumStep is not None: self.MaxNumStep = MaxNumStep
        if EndTime is not None: self.EndTime = EndTime
        
        if StopEnergyTol is not None: self.StopEnergyTol = StopEnergyTol
        self.StopEnergyType = StopEnergyType
        if StopMTol is not None: self.StopMTol = StopMTol
        if StopMxhTol is not None: self.StopMxhTol = StopMxhTol
        #######################
        if DataName is not None: self.DataName = DataName
        self.StepFigurePer = StepFigurePer
        if TotalFigurePer is not None: self.TotalFigurePer = TotalFigurePer
        #######################
        if Ku is not None: self.Ku = Ku
        if AnisotropyDirection is not None: self.AnisotropyDirection = AnisotropyDirection
        if He is not None: self.He = He
        if D is not None: self.D = D
        self.mu0 = mu0
        #################################
        if ATExample is not None: self.ATExample = ATExample
        self.ATExampleType = ATExampleType
        self.IfSpaceScale = IfSpaceScale
        self.BCSetting = BCSetting

        # Save some basic message.    
        if FileName is not None:
            MarkBasicImfor = self.__dict__
            np.savez(FileName, **MarkBasicImfor)

        ###############################
        ###############################
        if FileName is not None: 
            self.FileName = FileName
        else:
            self.FileName = None
        
        if MaxNumStep is None: self.MaxNumStep = None
        if StopEnergyTol is None: self.StopEnergyTol = None
        if StopMTol is None: self.StopMTol = None
        if StopMxhTol is None: self.StopMxhTol = None
        #######################
        if DataName is None: self.DataName = None
        #######################
        if Ku is None: self.Ku = None
        if AnisotropyDirection is None: self.AnisotropyDirection = None
        if He is None: self.He = None
        if D is None: self.D = None
        #################################
        if ATExample is None: self.ATExample = None


        ###############################################
        ###############################################
        self.FileName = FileName
        if self.IfSpaceScale == True:
            self.Lxyz = min(SpaceDomain[1] - SpaceDomain[0], SpaceDomain[3] - SpaceDomain[2], SpaceDomain[5] - SpaceDomain[4])
        else:
            self.Lxyz = 1
        self.SpaceDomain = np.zeros(shape = len(SpaceDomain), dtype = np.float64)
        for i in range(self.SpaceDomain.shape[0]): self.SpaceDomain[i] = SpaceDomain[i]/self.Lxyz
        #######################
        # Prepare for Node
        self.Deltat = (TimeDomain[1]  - TimeDomain[0])/ t
        self.Deltax = (self.SpaceDomain[1] - self.SpaceDomain[0])/ Nx
        self.Deltay = (self.SpaceDomain[3] - self.SpaceDomain[2])/ Ny
        self.Deltaz = (self.SpaceDomain[5] - self.SpaceDomain[4])/ Nz
        self.NC =  Nx* Ny* Nz
        self.Volume = (SpaceDomain[1] - SpaceDomain[0]) * (SpaceDomain[3] - SpaceDomain[2]) * (SpaceDomain[5] - SpaceDomain[4])
        self.VolumeOfUnitCell = self.Deltax * self.Deltay * self.Deltaz
         

        if EndTime is not None:
            self.MaxNumStep = int((EndTime - self.TimeDomain[0])/self.Deltat)
            if TotalFigurePer is not None: self.StepFigurePer = int(self.MaxNumStep/TotalFigurePer)
        else:
            self.MaxNumStep = MaxNumStep

        ##############################
        # Other Parameters
        self.Kd = self.mu0* self.Ms**2/2

        if self.ATExample is not None: self.RunAccuracyTestExample()
        

        self.ComputeCellCenterNode()
        self.MarkBdCell()
        self.PrepareFormnTimes()
        
        if 0 in TypeEffective: 
            self.CofOfAnisotropy = 2*self.Ku/self.mu0/self.Ms**2
            # self.WWP = np.sqrt(Cex/Ku)
            self.MatrixOfAnisotropyDirection()
        if 1 in TypeEffective: 
            self.CofOfExchange = 2*self.Cex/self.mu0/self.Ms**2/self.Lxyz**2
            self.SFEL = np.sqrt(Cex/self.Kd)
            self.ComputeMatrixOfDelta() 
        if 2 in TypeEffective: self.ComputeDemagnetizationTensor() # Preparing Work
        if 4 in TypeEffective: 
            self.CofOfIDMI = 2*self.D/self.mu0/self.Ms**2/self.Lxyz
            self.ComputeMatrixOfIDMI()

    def ComputeCellCenterNode(self): 
        xv = self.SpaceDomain[0] + np.arange(self.Nx, dtype = np.float64)*self.Deltax + self.Deltax/2
        yv = self.SpaceDomain[2] + np.arange(self.Ny, dtype = np.float64)*self.Deltay + self.Deltay/2
        zv = self.SpaceDomain[4] + np.arange(self.Nz, dtype = np.float64)*self.Deltaz + self.Deltaz/2
        # xv = np.linspace(self.SpaceDomain[0] + self.Deltax/2, self.SpaceDomain[1] - self.Deltax/2, self.Nx, dtype = np.float64)
        # yv = np.linspace(self.SpaceDomain[2] + self.Deltay/2, self.SpaceDomain[3] - self.Deltay/2, self.Ny, dtype = np.float64)
        # zv = np.linspace(self.SpaceDomain[4] + self.Deltaz/2, self.SpaceDomain[5] - self.Deltaz/2, self.Nz, dtype = np.float64)
        Value = np.zeros((self.Nx, self.Ny, self.Nz, 3), dtype = np.float64)
        Value[..., 0], Value[..., 1], Value[..., 2] = np.meshgrid(xv, yv, zv, indexing = "ij")
        self.CellCenterNode = Value.reshape(self.NC, 3)
        return True

    def MarkBdCell(self):
        Mark = np.arange( self.NC, dtype = int)

        self.isBdCellxDown = np.zeros(self.NC, dtype = bool)
        self.isBdCellxDown[0: self.Ny * self.Nz] = True

        self.isBdCellxUp = np.zeros(self.NC, dtype = bool)
        self.isBdCellxUp[- self.Ny * self.Nz:] = True
        self.isBdCellx = self.isBdCellxUp + self.isBdCellxDown

        self.bdCellxDown = Mark[self.isBdCellxDown]
        self.bdCellxUp = Mark[self.isBdCellxUp]
        self.bdCellx = Mark[self.isBdCellx]

        self.InvbdCellxDown = Mark[~self.isBdCellxDown]
        self.InvbdCellxUp = Mark[~self.isBdCellxUp]
        self.InvbdCellx = Mark[~self.isBdCellx]
        

        isBdCellyDown = np.zeros( self.Ny * self.Nz, dtype = bool)
        isBdCellyDown[0:self.Nz] = True
        isBdCellyDown = np.broadcast_to(isBdCellyDown[None, :], shape = (self.Nx, self.Ny * self.Nz)).reshape(-1)

        isBdCellyUp = np.zeros( self.Ny * self.Nz, dtype = bool)
        isBdCellyUp[-self.Nz:] = True
        isBdCellyUp = np.broadcast_to(isBdCellyUp[None, :], shape = (self.Nx, self.Ny * self.Nz )).reshape(-1)
        
        self.isBdCellyDown = isBdCellyDown
        self.isBdCellyUp = isBdCellyUp
        self.isBdCelly = self.isBdCellyDown + self.isBdCellyUp

        self.bdCellyDown = Mark[self.isBdCellyDown]
        self.bdCellyUp = Mark[self.isBdCellyUp]
        self.bdCelly = Mark[self.isBdCelly]

        self.InvbdCellyDown = Mark[~self.isBdCellyDown]
        self.InvbdCellyUp = Mark[~self.isBdCellyUp]
        self.InvbdCelly = Mark[~self.isBdCelly]


        isBdCellzDown = np.zeros( self.Nz, dtype = bool)
        isBdCellzDown[0] = True
        isBdCellzDown = np.broadcast_to(isBdCellzDown[None, None, :], shape = (self.Nx, self.Ny, self.Nz)).reshape(-1)


        isBdCellzUp = np.zeros(  self.Nz, dtype = bool)
        isBdCellzUp[-1] = True
        isBdCellzUp = np.broadcast_to(isBdCellzUp[None, None, :], shape = (self.Nx, self.Ny, self.Nz)).reshape(-1)
        
        self.isBdCellzDown = isBdCellzDown
        self.isBdCellzUp = isBdCellzUp
        self.isBdCellz = self.isBdCellzDown + self.isBdCellzUp

        self.bdCellzDown = Mark[self.isBdCellzDown]
        self.bdCellzUp = Mark[self.isBdCellzUp]
        self.bdCellz = Mark[self.isBdCellz]

        self.InvbdCellzDown = Mark[~self.isBdCellzDown]
        self.InvbdCellzUp = Mark[~self.isBdCellzUp]
        self.InvbdCellz = Mark[~self.isBdCellz]
        return True

    def Identity(self):
        Vlaue = np.ones(3*self.NC, dtype = np.float64)
        I = np.arange(3*self.NC, dtype = np.float64)
        J = np.arange(3*self.NC, dtype = np.float64)
        E = csr_matrix((Vlaue,(I,J)), shape=(3*self.NC, 3*self.NC))
        return E

    def OutProduct(self, A, B): # A = (3, N)
        LinA, LinB = A.reshape(3, -1), B.reshape(3, -1)
        C = np.zeros(LinA.shape, dtype = np.float64)
        C[0, :] = LinA[1, :] * LinB[2, :] - LinA[2, :] * LinB[1, :]
        C[1, :] = LinA[2, :] * LinB[0, :] - LinA[0, :] * LinB[2, :]
        C[2, :] = LinA[0, :] * LinB[1, :] - LinA[1, :] * LinB[0, :]
        return C.reshape(-1)

    def PrepareFormnTimes(self,):
        LinMark = np.arange(3*self.NC, dtype = int).reshape(3, self.NC)
        self.TimeI = np.concatenate((LinMark[2, :], LinMark[0, :], LinMark[1, :], LinMark[1, :], LinMark[2, :], LinMark[0, :]))
        self.TimeJ = np.concatenate((LinMark[1, :], LinMark[2, :], LinMark[0, :], LinMark[2, :], LinMark[0, :], LinMark[1, :]))
        return True
    
    def FunMatrixofmnTimes(self, mn):
        Value = np.append(mn, -mn)
        Matrix = csr_matrix((Value,(self.TimeI, self.TimeJ)), shape=(3*self.NC, 3*self.NC))
        return Matrix

    def L2Integral(self, m0, m1):
        return np.dot(m0, m1)*self.VolumeOfUnitCell
    
    def L2IntegralReal(self, m0, m1):
        return self.Lxyz**3 * self.VolumeOfUnitCell * np.dot(m0, m1)
    





########################################################################################
# Compute Effective Field
    def ComputeHExchange(self, m):
        # return 2*self.Cex/self.Ms**2 * self.FullMatrix_Delta@m
        return self.CofOfExchange * (self.FullMatrixOfDelta@m) 

    # def ComputeHAnisotropyOld(self, m):
    #     Lin = np.einsum('i, ij->j', self.AnisotropyDirection, m.reshape(3, self.NC)) * self.AnisotropyDirection.reshape(3, 1)
    #     return 2*self.Ku/self.Ms**2 * Lin.reshape(-1)
    
    def ComputeHAnisotropy(self, m):
        return self.CofOfAnisotropy * (self.MatrixOfAnisotropy@m)

    def ComputeHExternal(self, tn):
        return self.He(self.CellCenterNode, tn = tn)

    # def EffectiveField(self, m, tn = None, FullShow = False):
    #     HAni = HEx = Hs = He = np.zeros(3*self.NC, dtype = np.float64)
    #     if 0 in self.TypeEffective: HAni = self.ComputeHAnisotropy(m = m)
    #     if 1 in self.TypeEffective: HEx = self.ComputeHExchange(m = m)
    #     if 2 in self.TypeEffective: Hs = self.Computemu0Hs(m = m)
    #     if 3 in self.TypeEffective: He = self.ComputeHExternal(tn = tn)
    #     Value = HAni + HEx + Hs + He
    #     if FullShow == False: return Value
    #     else: return Value, HAni, HEx, Hs, He


    #def ComputeHIDMI(self, m):
    #    return self.CofOf IDMI* (self.FullMatrixOfIDMI@m)

    def EffectiveField(self, m, tn = None, FullShow = False):
        HEach = np.zeros( (len(self.TypeEffective), 3*self.NC), dtype = np.float64)
        Mark = 0
        if 0 in self.TypeEffective: 
            HEach[Mark, :] = self.ComputeHAnisotropy(m = m)
            Mark += 1
        if 1 in self.TypeEffective: 
            HEach[Mark, :]  = self.ComputeHExchange(m = m)
            Mark += 1
        if 2 in self.TypeEffective: 
            HEach[Mark, :]  = self.Computemu0Hs(m = m) # Note, 传入m ，退磁
            Mark += 1
        if 3 in self.TypeEffective: 
            HEach[Mark, :]  = self.ComputeHExternal(tn = tn)
            Mark += 1
        if 4 in self.TypeEffective: 
            HEach[Mark, :]  = self.ComputeHIDMI(m = m)
        Value = np.sum(HEach, axis = 0)
        if FullShow == False: return Value
        else: return Value, HEach


    # def EffectiveField(self, m, tn = None, FullShow = False):
    #     AllH = np.zeros((len(self.TypeEffective), 3*self.NC), dtype = np.float64)
    #     Lini = 0
    #     if 0 in self.TypeEffective: 
    #         AllH[Lini, :] = self.ComputeHAnisotropy(m = m)
    #         lini += 1
    #     if 1 in self.TypeEffective: 
    #         AllH[Lini, :] = self.ComputeHExchange(m = m)
    #         lini += 1
    #     if 2 in self.TypeEffective: 
    #         AllH[Lini, :] = self.Computemu0Hs(m = m)
    #         lini += 1
    #     if 3 in self.TypeEffective: 
    #         AllH[Lini, :] = self.ComputeHExternal(tn = tn)
    #     TotalH = np.sum(H, axis = 0)
    #     if FullShow == False: return TotalH
    #     else: return TotalH, H



########################################################################################
# Compute Energy
    def ExternalEnergy(self, m, tn = None, HExternal = None):
        if HExternal is None: HExternal = self.ComputeHExternal(tn)
        return - self.Lxyz**3 * self.mu0 * self.Ms**2 * self.VolumeOfUnitCell * np.dot(m, HExternal)

    # def AnisotropyEnergyOld(self, m0): # m0 = (3, NC)
    #     Lin = np.einsum('i, ij->j', self.AnisotropyDirection, m0.reshape(3, self.NC))
    #     Lin = ( np.einsum('ij, ij->j', m0.reshape(3, self.NC), m0.reshape(3, self.NC)) - Lin**2)
    #     E0 = self.Ku/self.Ms**2 * self.VolumeOfUnitCell * np.sum(Lin)
    #     return E0
    def AnisotropyEnergy(self, m, HAnisotropy = None): # m0 = (3, NC)
        if HAnisotropy is None: HAnisotropy = self.ComputeHAnisotropy(m = m)
        return self.Lxyz**3 * self.VolumeOfUnitCell  *(self.Ku * self.NC - self.Ms**2 /2 * self.mu0 * np.dot(HAnisotropy, m))
        # return self.Lxyz*3 * self.VolumeOfUnitCell  *(self.Ku * np.dot(m, m) - self.Ms**2 /2 * self.mu0 * np.dot(HAnisotropy, m))
    
    
    # def ExchangeEnergy(self, m0):
    #     Value = np.einsum('ij, ij->', ( self.FullMatrixOfDelta@m0).reshape(3, self.NC), m0.reshape(3, self.NC))
    #     E1 = - self.Cex / self.Ms**2 * self.VolumeOfUnitCell * Value
    #     return E1
    def ExchangeEnergy(self, m, HExchange = None):
        if HExchange is None: HExchange = self.ComputeHExchange(m = m)
        return -  self.Lxyz**3 * self.VolumeOfUnitCell * self.Ms**2 *self.mu0 /2* np.dot(HExchange, m) 

    # def DiscreteEnergy(self, m, tn = None, HAnisotropy = None, HExchange = None, HStrayField = None, HExternal = None):
    #     E0, E1, E2, E3 = 0, 0, 0, 0
    #     if 0 in self.TypeEffective: E0 = self.AnisotropyEnergy(m = m, HAnisotropy = HAnisotropy)
    #     if 1 in self.TypeEffective: E1 = self.ExchangeEnergy(m = m, HExchange = HExchange)
    #     if 2 in self.TypeEffective: E2 = self.StrayFieldEnergy(m = m, HStrayField = HStrayField)
    #     if 3 in self.TypeEffective: E3 = self.ExternalEnergy(m = m, tn = tn, HExternal = HExternal)
    #     E = E0 + E1 + E2 + E3
    #     return E, E0, E1, E2, E3

    def IDMIEnergy(self, m, HIDMI = None):
        if HIDMI is None: HIDMI = self.ComputeHIDMI(m = m)
        return - self.Lxyz**3 * self.VolumeOfUnitCell * self.Ms**2 *self.mu0 /2* np.dot(HIDMI, m) 

    def DiscreteEnergy(self, m, tn = None):
        EEach = np.zeros(len(self.TypeEffective), dtype = np.float64)
        Mark = 0
        if 0 in self.TypeEffective: 
            EEach[Mark] = self.AnisotropyEnergy(m = m)
            Mark += 1
        if 1 in self.TypeEffective: 
            EEach[Mark] = self.ExchangeEnergy(m = m)
            Mark += 1
        if 2 in self.TypeEffective: 
            EEach[Mark] = self.StrayFieldEnergy(m = m)
            Mark += 1
        if 3 in self.TypeEffective: 
            EEach[Mark] = self.ExternalEnergy(m = m, tn = tn)
            Mark += 1
        if 4 in self.TypeEffective: 
            EEach[Mark] = self.IDMIEnergy(m = m)
        E = np.sum(EEach)
        return [E] + [EEach[i] for i in range(len(self.TypeEffective))]

    def DiscreteEnergyWithH(self, m, HEach):
        EEach = np.zeros(len(self.TypeEffective), dtype = np.float64)
        Mark = 0
        if 0 in self.TypeEffective: 
            EEach[Mark] = self.AnisotropyEnergy(m = m, HAnisotropy = HEach[Mark, :])
            Mark += 1
        if 1 in self.TypeEffective: 
            EEach[Mark] = self.ExchangeEnergy(m = m, HExchange = HEach[Mark, :])
            Mark += 1
        if 2 in self.TypeEffective: 
            EEach[Mark] = self.StrayFieldEnergy(m = m, HStrayField = HEach[Mark, :])
            Mark += 1
        if 3 in self.TypeEffective: 
            EEach[Mark] = self.ExternalEnergy(m = m, HExternal = HEach[Mark, :])
            Mark += 1
        if 4 in self.TypeEffective: 
            EEach[Mark] = self.IDMIEnergy(m = m, HIDMI = HEach[Mark, :])
        E = np.sum(EEach)
        return [E] + [EEach[i] for i in range(len(self.TypeEffective))]
    
    
    
########################################################################################
    def OOMMFToPython(self, m):
        m = m.reshape(self.Nz, self.Ny, self.Nx, 3) 
        return np.transpose(m, (3, 2, 1, 0)).reshape(-1)

    def PythonToOOMMF(self, m, FileName = None, fmt = None):
        Value = m.reshape(3, self.Nx, self.Ny, self.Nz)
        Value = np.transpose(Value, (3, 2, 1, 0)).reshape(self.NC, 3)
        if FileName is not None: np.savetxt(FileName, Value, fmt = fmt)
        return Value
    
    # def MToOVF(self, m, Renormlize = False, Decimals = 11):
    #     if Renormlize == False: MagOfM = 1
    #     else: MagOfM = np.linalg.norm(m.reshape(3, self.Nx, self.Ny, self.Nz), axis = 0)
    #     Linm = (m.reshape(3, self.Nx, self.Ny, self.Nz)/MagOfM).round(decimals = Decimals)
    #     Linm = np.transpose(Linm, (3, 2, 1, 0)).reshape(self.NC, 3)
    #     Path = "C:/Users/zjjis/Desktop/"
    #     with open(Path + "M_Field.ovf", "w") as file:
    #         file.write("# OOMMF: rectangular mesh v1.0\n")
    #         file.write("# Segment count: 1\n")
    #         file.write("# Begin: Segment\n")
    #         file.write("# Begin: Header\n")
    #         file.write("# Title: M-Field\n")
    #         file.write("# meshtype: rectangular\n")
    #         file.write("# meshunit: m\n")
    #         file.write(f"# xstepsize: {self.Deltax}\n")
    #         file.write(f"# ystepsize: {self.Deltay}\n")
    #         file.write(f"# zstepsize: {self.Deltaz}\n")
    #         file.write(f"# xnodes: {self.Nx}\n")
    #         file.write(f"# ynodes: {self.Ny}\n")
    #         file.write(f"# znodes: {self.Nz}\n")
    #         file.write("# End: Header\n")
    #         file.write("# Begin: Data Text\n")
    #         for i in range(self.NC): file.write(f"{Linm[i][0]} {Linm[i][1]} {Linm[i][2]}\n")
    #         file.write("# End: Data Text\n")
    #         file.write("# End: Segment\n")
    #     return True

    # def MToOVF(self, m, Renormlize = False, Decimals = 11):
    #     if Renormlize == False: MagOfM = 1
    #     else: MagOfM = np.linalg.norm(m.reshape(3, self.Nx, self.Ny, self.Nz), axis = 0)
    #     Linm = m.reshape(3, self.Nx, self.Ny, self.Nz)/MagOfM
    #     Linm = np.transpose(Linm, (3, 2, 1, 0)).reshape(self.NC, 3)
    #     Path = "C:/Users/zjjis/Desktop/"
    #     with open(Path + "M_Field.ovf", "w") as file:
    #         file.write("# OOMMF: rectangular mesh v1.0\n")
    #         file.write("# Segment count: 1\n")
    #         file.write("# Begin: Segment\n")
    #         file.write("# Begin: Header\n")
    #         file.write("# Title: M-Field\n")
    #         file.write("# meshtype: rectangular\n")
    #         file.write("# meshunit: m\n")
    #         file.write(f"# xstepsize: {self.Deltax}\n")
    #         file.write(f"# ystepsize: {self.Deltay}\n")
    #         file.write(f"# zstepsize: {self.Deltaz}\n")
    #         file.write(f"# xnodes: {self.Nx}\n")
    #         file.write(f"# ynodes: {self.Ny}\n")
    #         file.write(f"# znodes: {self.Nz}\n")
    #         file.write("# End: Header\n")
    #         file.write("# Begin: Data Text\n")
    #         for i in range(self.NC): file.write(f"{Linm[i][0]:.11f} {Linm[i][1]:.11f} {Linm[i][2]:.11f}\n")
    #         file.write("# End: Data Text\n")
    #         file.write("# End: Segment\n")
    #     return True

    # def MToOVF(self, m, Renormlize = True, Decimals = 11):
    #     if Renormlize == False: MagOfM = 1
    #     else: MagOfM = np.linalg.norm(m.reshape(3, self.Nx, self.Ny, self.Nz), axis = 0)
    #     Linm = m.reshape(3, self.Nx, self.Ny, self.Nz)/MagOfM
    #     Linm = np.transpose(Linm, (3, 2, 1, 0)).reshape(self.NC, 3)
    #     Path = "D:/OneDrive/Documents/LearningAndTeaching/Python/Micromagnetics_FDM/"
    #     with open(Path + "M_Field.ovf", "wb", encoding='utf-8') as file:
    #         file.write("# OOMMF OVF 2.0\n")
    #         file.write("# Segment count: 1\n")
    #         file.write("# Begin: Segment\n")
    #         file.write("# Begin: Header\n")
    #         file.write("# Title: m\n")
    #         file.write("# meshtype: rectangular\n")
    #         file.write("# meshunit: m\n")
    #         file.write(f"# xmin: {self.SpaceDomain[0]*self.Lxyz}\n")
    #         file.write(f"# ymin: {self.SpaceDomain[2]*self.Lxyz}\n")
    #         file.write(f"# zmin: {self.SpaceDomain[4]*self.Lxyz}\n")
    #         file.write(f"# xmax: {self.SpaceDomain[1]*self.Lxyz}\n")
    #         file.write(f"# ymax: {self.SpaceDomain[3]*self.Lxyz}\n")
    #         file.write(f"# zmax: {self.SpaceDomain[5]*self.Lxyz}\n")
    #         file.write("# valuedim: 3\n")
    #         file.write("# valuelabels: m_x m_y m_z\n")
    #         file.write("# valueunits: 1 1 1\n")
    #         file.write("# Desc: Total simulation time:  4.099999999999999e-08  s\n")
    #         file.write(f"# xbase: {self.Deltax*self.Lxyz/2}\n")
    #         file.write(f"# ybase: {self.Deltay*self.Lxyz/2}\n")
    #         file.write(f"# zbase: {self.Deltaz*self.Lxyz/2}\n")
    #         file.write(f"# xnodes: {self.Nx}\n")
    #         file.write(f"# ynodes: {self.Ny}\n")
    #         file.write(f"# znodes: {self.Nz}\n")
    #         file.write(f"# xstepsize: {self.Deltax*self.Lxyz}\n")
    #         file.write(f"# ystepsize: {self.Deltay*self.Lxyz}\n")
    #         file.write(f"# zstepsize: {self.Deltaz*self.Lxyz}\n")
    #         file.write("# End: Header\n")
    #         file.write("# Begin: Data Text\n")
    #         for i in range(self.NC): file.write(f"{Linm[i][0]:.11f} {Linm[i][1]:.11f} {Linm[i][2]:.11f}\n")
    #         file.write("# End: Data Text\n")
    #         file.write("# End: Segment\n")
    #     return True



    def MToOVF(self, m, Renormlize=True, Decimals=11, Path = None):
        if not Renormlize:
            MagOfM = 1
        else:
            MagOfM = np.linalg.norm(m.reshape(3, self.Nx, self.Ny, self.Nz), axis=0)
        Linm = m.reshape(3, self.Nx, self.Ny, self.Nz) / MagOfM
        Linm = np.transpose(Linm, (3, 2, 1, 0)).reshape(self.NC, 3)
        if Path is None: Path = "D:/OneDrive/Documents/LearningAndTeaching/Python/Micromagnetics_FDM/M_Field.ovf"
        
        with open(Path, "wb") as file:
            header = (
                "# OOMMF OVF 2.0\n"
                "# Segment count: 1\n"
                "# Begin: Segment\n"
                "# Begin: Header\n"
                "# Title: m\n"
                "# meshtype: rectangular\n"
                "# meshunit: m\n"
                f"# xmin: {self.SpaceDomain[0]*self.Lxyz}\n"
                f"# ymin: {self.SpaceDomain[2]*self.Lxyz}\n"
                f"# zmin: {self.SpaceDomain[4]*self.Lxyz}\n"
                f"# xmax: {self.SpaceDomain[1]*self.Lxyz}\n"
                f"# ymax: {self.SpaceDomain[3]*self.Lxyz}\n"
                f"# zmax: {self.SpaceDomain[5]*self.Lxyz}\n"
                "# valuedim: 3\n"
                "# valuelabels: m_x m_y m_z\n"
                "# valueunits: 1 1 1\n"
                "# Desc: Total simulation time:  4.099999999999999e-08  s\n"
                f"# xbase: {self.Deltax*self.Lxyz/2}\n"
                f"# ybase: {self.Deltay*self.Lxyz/2}\n"
                f"# zbase: {self.Deltaz*self.Lxyz/2}\n"
                f"# xnodes: {self.Nx}\n"
                f"# ynodes: {self.Ny}\n"
                f"# znodes: {self.Nz}\n"
                f"# xstepsize: {self.Deltax*self.Lxyz}\n"
                f"# ystepsize: {self.Deltay*self.Lxyz}\n"
                f"# zstepsize: {self.Deltaz*self.Lxyz}\n"
                "# End: Header\n"
                "# Begin: Data Text\n"
            )
            file.write(header.encode('utf-8'))

            for i in range(self.NC):
                line = f"{Linm[i][0]:.11f} {Linm[i][1]:.11f} {Linm[i][2]:.11f}\n"
                file.write(line.encode('utf-8'))

            footer = (
                "# End: Data Text\n"
                "# End: Segment\n"
            )
            file.write(footer.encode('utf-8'))
        
        return True

########################################################################################
# Judge for Break
    def JudgeIfBreakOnM(self, m0, m1):
        # Recommand StopMTol = 1e-5
        # Value = np.max(np.linalg.norm(((m0 - m1)/self.Deltat).reshape(3, self.NC), axis =0))
        # Value = np.max(np.linalg.norm((m0 - m1).reshape(3, self.NC), axis =0))
        Value = np.max(np.linalg.norm((m0 - m1).reshape(3, self.NC), axis =0))
        print("Difference on m0 and m1", Value)
        if Value < self.StopMTol: return False
        return True
    
    def JudgeIfBreakOnEnergy(self, Energy0, Energy1):
        Value = np.abs((Energy0[self.StopEnergyType] - Energy1[self.StopEnergyType]) / Energy0[self.StopEnergyType])
        print('Difference on Energy of m0 and m1', Value)
        if Value < self.StopEnergyTol: return False
        return True

    def JudgeIfBreakOnMxh(self, m1, Hm1 = None, tn = None):
        if Hm1 is None: Hm1 = self.EffectiveField(m0 = m1, tn = tn)
        Value = np.max(np.linalg.norm((self.OutProduct(m1, Hm1)).reshape(3, self.NC), axis = 0))
        print("Value of Mxh", Value)
        if Value < self.StopMxhTol: return False
        return True

    def MeanOfM(self, m):
        return np.mean(m.reshape(3, self.NC), axis = -1)/self.Ms

    def RenormalizeOfM(self, m):
        MagOfM = np.linalg.norm(m.reshape(3, self.NC), axis = 0)
        return (m.reshape(3, self.NC)/MagOfM).reshape(-1)

    ############################
    # Compute Initial m1, m2 by Runge-Kutta methods.
#     def RungeKuttaForK(self, m, tn = None):
#         LinH = self.EffectiveField(m, tn = tn)
#         mTH = self.OutProduct(m, LinH)
#         K = - 1 / (1+ self.alpha**2) * mTH  \
#             - self.alpha / (1+ self.alpha**2) * self.OutProduct(m, mTH)
#         if self.ForcingItem is not None:
#             ValueForcing = self.ForcingItem(p = self.CellCenterNode, tn = tn)
#             K += self.alpha / (1 + self.alpha**2)/self.Ms * self.OutProduct(m, ValueForcing) \
#                 + ValueForcing.reshape(-1)/ (1 + self.alpha**2)
#         return K
              
#     def RungeKuttaForm1(self, m0, tn = None):
#         Deltat = self.Deltat
#         if tn is None: tn  = self.TimeDomain[0]
#         K1 = self.RungeKuttaForK(m0, tn)
#         K2 = self.RungeKuttaForK(m0 + Deltat /2 *K1, tn + Deltat /2)
#         K3 = self.RungeKuttaForK(m0 + Deltat /2 *K2, tn + Deltat /2)
#         K4 = self.RungeKuttaForK(m0 + Deltat * K3, tn + Deltat)
#         return m0 + Deltat * (1/6 * K1 + 1/3 * K2 + 1/3 * K3 + 1/6 * K4)

    ############################
    # Compute Initial M1, M1 by Runge-Kutta methods.
    def PrepareRungeKuttaForLL(self, m, tn = None):
        LinH = self.EffectiveField(m, tn = tn)
        mTH = self.OutProduct(m, LinH)
        K = - mTH - self.alpha * self.OutProduct(m, mTH)
        # if self.ForcingItem is not None:
        #     ValueForcing = self.ForcingItem(p = self.CellCenterNode, tn = tn, alpha = self.alpha, gamma = self.gamma, Ms = self.Ms)
        #     K += self.alpha / (1 + self.alpha**2)/self.Ms * self.OutProduct(m, ValueForcing) \
        #         + ValueForcing.reshape(-1)/ (1 + self.alpha**2)
        return K
              
    def RungeKuttaForLL(self, m0, tn = None):
        Deltat = self.Deltat
        if tn is None: tn  = self.TimeDomain[0]
        K1 = self.PrepareRungeKuttaForLL(m0, tn)
        K2 = self.PrepareRungeKuttaForLL(m0 + Deltat /2 *K1, tn + Deltat /2)
        K3 = self.PrepareRungeKuttaForLL(m0 + Deltat /2 *K2, tn + Deltat /2)
        K4 = self.PrepareRungeKuttaForLL(m0 + Deltat * K3, tn + Deltat)
        return m0 + Deltat * (1/6 * K1 + 1/3 * K2 + 1/3 * K3 + 1/6 * K4)


    def InitialM(self, IMType = "Diamond", ROfDMI = None):
        m0 = np.zeros((3, self.Nx, self.Ny, self.Nz), dtype = np.float64)
        if IMType == "Diamond":
            m0[0, 0:int((self.Nx)/2), 0:int((self.Ny)/2), :] = -1
            m0[0, 0:int((self.Nx)/2), int((self.Ny)/2):, :] = 1
            m0[0, int((self.Nx)/2):, 0:int((self.Ny)/2), :] = 1
            m0[0, int((self.Nx)/2):, int((self.Ny)/2):, :] = -1
        if IMType == "Diamond2":
            m0[1, 0:int((self.Nx)/4), :, :] = 1
            m0[1, int((self.Nx)/4):int((self.Nx)*3/4), :, :] = -1
            m0[1, int((self.Nx)*3/4):, :, :] = 1
        if IMType == "Landau" or IMType == "SCT":
            m0[0, :, 0:int((self.Ny)/2), :] = 1
            m0[0, :, int((self.Ny)/2):, :] = -1
        if IMType == "DCT":
            m0[1, 0:int(self.Nx/4), :, :] = 1
            m0[1, int(self.Nx/4):int(3*self.Nx/8), :, :] = -1
            m0[1, int(3*self.Nx/8):int(4*self.Nx/8), :, :] = 1
            m0[1, int(4*self.Nx/8):int(5*self.Nx/8), :, :] = -1
            m0[1, int(5*self.Nx/8):int(6*self.Nx/8), :, :] = 1
            m0[1, int(6*self.Nx/8):, :, :] = -1
        if IMType == "S":
            m0[0, :] = 1/np.sqrt(1.01)
            m0[1, :] = 0.1/np.sqrt(1.01)
        if IMType == "S2":
            m0[0, :] = 0.8/np.sqrt(0.8**2 + 0.2**2)
            m0[1, :] = 0.2/np.sqrt(0.8**2 + 0.2**2)
        if IMType == "Flower":
            m0[0, :] = 1
        if IMType == "DMI":
            m0[2, :] = 1
            LinCentralNode = np.array([(self.SpaceDomain[1] + self.SpaceDomain[0])/2, (self.SpaceDomain[3] + self.SpaceDomain[2])/2, (self.SpaceDomain[5] + self.SpaceDomain[4])/2], dtype = np.float64)
            # Mark = (np.sqrt(np.einsum('ij, ij->i', self.CellCenterNode - LinCentralNode, self.CellCenterNode - LinCentralNode)) <= ROfDMI)
            # print('shape', self.CellCenterNode.shape, LinCentralNode.shape)
            # print('self.CellCenterNode', self.CellCenterNode)
            # print('LinCentralNode', LinCentralNode)
            Mark = (np.linalg.norm(self.CellCenterNode - LinCentralNode, axis = -1) <= ROfDMI/self.Lxyz)
            m0 = m0.reshape(3, -1)
            m0[2, Mark] = -1
        return m0.reshape(-1)



'''
Here we provide the "Nabla in Cell" and present another computation of "ExchangeEnergy" based on "Nabla in Cell". 

    def PrepareForComputeNablaInCell(self, xyzType, DirectionType, bdCellxUp):
        if xyzType == "x":
            LinNxyz = self.Ny*self.Nz
        elif xyzType == "y":
            LinNxyz = self.Nz
        elif xyzType == "z":
            LinNxyz = 1
        DirType = 1 if DirectionType == "Up" else -1
        J = np.arange(self.NC, dtype = int).reshape(self.NC, 1) + np.arange(2, dtype = int) * LinNxyz * DirType
        # J[bdCellxUp, -1] = J[bdCellxUp, -1] - 2 * LinNxyz * DirType
        J[bdCellxUp, -1] = J[bdCellxUp, -1] - LinNxyz * DirType
        return J

    def ComputeNablaInCell(self):
        Value = np.array([1, -1], dtype = np.float64)
        Value = np.broadcast_to(Value[None, :], shape = (self.NC, 2))
        I = np.broadcast_to(np.arange(self.NC, dtype = int)[:, None], shape = (self.NC, 2))
        

        J = self.PrepareForComputeNablaInCell('x', 'Down', self.bdCellxDown)
        self.NablaxDownCell = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(self.NC, self.NC)) / self.Deltax
        J = self.PrepareForComputeNablaInCell('x', 'Up', self.bdCellxUp)
        self.NablaxUpCell = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(self.NC, self.NC)) / self.Deltax


        J = self.PrepareForComputeNablaInCell('y', 'Down', self.bdCellyDown)
        self.NablayDownCell = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(self.NC, self.NC)) / self.Deltay
        J = self.PrepareForComputeNablaInCell('y', 'Up', self.bdCellyUp)
        self.NablayUpCell = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(self.NC, self.NC)) / self.Deltay    


        J = self.PrepareForComputeNablaInCell('z', 'Down', self.bdCellzDown)
        self.NablazDownCell = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(self.NC, self.NC)) / self.Deltaz
        J = self.PrepareForComputeNablaInCell('z', 'Up', self.bdCellzUp)
        self.NablazUpCell = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(self.NC, self.NC)) / self.Deltaz

        return True

    def ExchangeEnergy(self, m0):
        Linm0 = m0.reshape(3, self.NC)
        Value = np.zeros((6, 3, self.NC), dtype = np.float64)
        for i in range(3):
            Value[0, i, :] = self.NablaxDownCell @ Linm0[i, :]  / self.Deltax   # (NC,)
            Value[1, i, :] = self.NablaxUpCell @ Linm0[i, :]  / self.Deltax     # (NC,)
            Value[2, i, :] = self.NablayDownCell @ Linm0[i, :] / self.Deltay    # (NC,)
            Value[3, i, :] = self.NablayUpCell @ Linm0[i, :]  / self.Deltay     # (NC,)
            Value[4, i, :] = self.NablazDownCell @ Linm0[i, :] / self.Deltaz    # (NC,)
            Value[5, i, :] = self.NablazUpCell @ Linm0[i, :] / self.Deltaz      # (NC,)
        E1 = self.Cex / self.Ms**2 * self.VolumeOfUnitCell *np.einsum('ijk, jk->', Value, Linm0)
        return E1
'''

