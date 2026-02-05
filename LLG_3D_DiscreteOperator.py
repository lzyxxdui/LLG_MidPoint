import numpy as np
from scipy.sparse import csr_matrix

class DiscreteOperator(object):
    
    
    def MatrixOfAnisotropyDirection(self):
        Value = self.AnisotropyDirection.reshape(1, 3) * self.AnisotropyDirection.reshape(3, 1) #(3, 3)
        PositionOfZeros = np.where(Value != 0)
        NumOfNonZero = len(PositionOfZeros[0])
        Value = np.broadcast_to(Value[PositionOfZeros][:, None], shape = (NumOfNonZero, self.NC))
        Mark, Mark2 = np.arange(3, dtype = int), np.arange(self.NC, dtype = int).reshape(1, self.NC)
        I = self.NC * np.broadcast_to(Mark[:, None], shape = (3, 3))[PositionOfZeros].reshape(NumOfNonZero, 1) + Mark2
        J = self.NC * np.broadcast_to(Mark[None, :], shape = (3, 3))[PositionOfZeros].reshape(NumOfNonZero, 1) + Mark2
        self.MatrixOfAnisotropy = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(3*self.NC, 3*self.NC))
        return True
    
    def PrepareForMatrixOfDelta(self, Mark, Mark2, InvbdCellx, Deltax):
        ValueInter = np.broadcast_to(Mark[None, :], shape = InvbdCellx.shape + Mark.shape) / Deltax**2
        IInter = np.broadcast_to(InvbdCellx[:, None], shape = InvbdCellx.shape + Mark.shape)
        # JInter = IInter + np.broadcast_to(Mark2[None, :], shape = InvbdCellx.shape + Mark2.shape)
        JInter = IInter + Mark2
        return ValueInter, IInter, JInter

    def ComputeMatrixOfDelta(self):
        Mark = np.array([1, -2, 1])
        Mark2 = np.array([- self.Ny * self.Nz, 0, self.Ny * self.Nz ], dtype = np.float64)
        if self.Nx in (1, ): 
            ValueInterx, IInterx, JInterx = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueInterx, IInterx, JInterx = self.PrepareForMatrixOfDelta(Mark, Mark2, self.InvbdCellx, self.Deltax)
        Mark2 = np.array([-  self.Nz , 0,  self.Nz ], dtype = np.float64)
        if self.Ny in (1, ): 
            ValueIntery, IIntery, JIntery = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueIntery, IIntery, JIntery = self.PrepareForMatrixOfDelta(Mark, Mark2, self.InvbdCelly, self.Deltay)
        Mark2 = np.array([- 1, 0, 1], dtype = np.float64)
        if self.Nz in (1, ): 
            ValueInterz, IInterz, JInterz = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueInterz, IInterz, JInterz = self.PrepareForMatrixOfDelta(Mark, Mark2, self.InvbdCellz, self.Deltaz)

            
        #############################################
        # Down
        if self.BCSetting == "niemann":
            Mark = np.array([-1, 1])
            Mark2 = np.array([0,  self.Ny * self.Nz ], dtype = np.float64)
        elif self.BCSetting == "periodic":
            Mark = np.array([1, -2, 1])
            Mark2 = np.array([(self.Nx - 1)*self.Ny * self.Nz, 0,  self.Ny * self.Nz], dtype = np.float64)

        if self.Nx in (1, ): 
            ValueDownx, IDownx, JDownx = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueDownx, IDownx, JDownx = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellxDown, self.Deltax)
        


        if self.BCSetting == "niemann":
            Mark2 = np.array([0,  self.Nz ], dtype = np.float64)
        elif self.BCSetting == "periodic":
            Mark2 = np.array([(self.Ny - 1)*self.Nz, 0,  self.Nz ], dtype = np.float64)

        if self.Ny in (1, ): 
            ValueDowny, IDowny, JDowny = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueDowny, IDowny, JDowny = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellyDown, self.Deltay)
        
        if self.BCSetting == "niemann":
            Mark2 = np.array([0, 1], dtype = np.float64)
        elif self.BCSetting == "periodic":
            Mark2 = np.array([self.Nz-1, 0, 1], dtype = np.float64)

        if self.Nz in (1, ): 
            ValueDownz, IDownz, JDownz = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueDownz, IDownz, JDownz = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellzDown, self.Deltaz)

        #############################################
        # Up
        if self.BCSetting == "niemann":
            Mark = np.array([1, -1])
            Mark2 = np.array([- self.Ny * self.Nz , 0], dtype = np.float64)
        elif self.BCSetting == "periodic":
            Mark = np.array([1, -2, 1])
            Mark2 = np.array([- self.Ny * self.Nz , 0, - (self.Nx - 1)*self.Ny * self.Nz], dtype = np.float64)

        if self.Nx in (1, ): 
            ValueUpx, IUpx, JUpx = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueUpx, IUpx, JUpx = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellxUp, self.Deltax)

        if self.BCSetting == "niemann":  
            Mark2 = np.array([- self.Nz , 0], dtype = np.float64)
        elif self.BCSetting == "periodic":
            Mark2 = np.array([- self.Nz , 0, - (self.Ny - 1)*self.Nz], dtype = np.float64)


        if self.Ny in (1, ): 
            ValueUpy, IUpy, JUpy = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueUpy, IUpy, JUpy = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellyUp, self.Deltay)

        if self.BCSetting == "niemann": 
            Mark2 = np.array([-1, 0], dtype = np.float64)
        elif self.BCSetting == "periodic":
            Mark2 = np.array([-1, 0, - self.Nz + 1], dtype = np.float64)
        if self.Nz in (1, ): 
            ValueUpz, IUpz, JUpz = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueUpz, IUpz, JUpz = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellzUp, self.Deltaz)

        Value = np.concatenate((ValueInterx.reshape(-1), ValueIntery.reshape(-1), ValueInterz.reshape(-1), \
                                ValueDownx.reshape(-1), ValueDowny.reshape(-1), ValueDownz.reshape(-1), \
                                ValueUpx.reshape(-1), ValueUpy.reshape(-1), ValueUpz.reshape(-1)))
        I = np.concatenate((IInterx.reshape(-1), IIntery.reshape(-1), IInterz.reshape(-1), \
                            IDownx.reshape(-1), IDowny.reshape(-1), IDownz.reshape(-1), \
                            IUpx.reshape(-1), IUpy.reshape(-1), IUpz.reshape(-1)))
        J = np.concatenate((JInterx.reshape(-1), JIntery.reshape(-1), JInterz.reshape(-1), \
                            JDownx.reshape(-1), JDowny.reshape(-1), JDownz.reshape(-1), \
                            JUpx.reshape(-1), JUpy.reshape(-1), JUpz.reshape(-1)))

        self.MatrixOfDelta = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(self.NC, self.NC))
        Value = np.concatenate((Value.reshape(-1), Value.reshape(-1), Value.reshape(-1)))
        I = np.concatenate((I.reshape(-1), I.reshape(-1) + self.NC, I.reshape(-1) + 2*self.NC))
        J = np.concatenate((J.reshape(-1), J.reshape(-1) + self.NC, J.reshape(-1) + 2*self.NC))            
        self.FullMatrixOfDelta = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(3*self.NC, 3*self.NC))
        return True
    
    

    def PrepareForMatrixOfIDMI(self, Mark, Mark2, InvbdCellx, Deltax):
        ValueInter = np.broadcast_to(Mark[None, :], shape = InvbdCellx.shape + Mark.shape) / (2*Deltax)
        IInter = np.broadcast_to(InvbdCellx[:, None], shape = InvbdCellx.shape + Mark.shape)
        JInter = IInter + Mark2
        return ValueInter, IInter, JInter

    def ComputeMatrixOfIDMI(self):
        Mark = np.array([-1, 1])
        Mark2 = np.array([- self.Ny * self.Nz, self.Ny * self.Nz ], dtype = np.float64)
        if self.Nx in (1, ): 
            ValueInterx, IInterx, JInterx = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueInterx, IInterx, JInterx = self.PrepareForMatrixOfIDMI(Mark, Mark2, self.InvbdCellx, self.Deltax)
        Mark2 = np.array([-  self.Nz , self.Nz ], dtype = np.float64)
        if self.Ny in (1, ): 
            ValueIntery, IIntery, JIntery = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueIntery, IIntery, JIntery = self.PrepareForMatrixOfIDMI(Mark, Mark2, self.InvbdCelly, self.Deltay)

        #############################################
        # Down
        Mark = np.array([-1, 1])
        Mark2 = np.array([0,  self.Ny * self.Nz ], dtype = np.float64)
        if self.Nx in (1, ): 
            ValueDownx, IDownx, JDownx = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueDownx, IDownx, JDownx = self.PrepareForMatrixOfIDMI(Mark, Mark2, self.bdCellxDown, self.Deltax)
            
        Mark2 = np.array([0,  self.Nz ], dtype = np.float64)
        if self.Ny in (1, ): 
            ValueDowny, IDowny, JDowny = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueDowny, IDowny, JDowny = self.PrepareForMatrixOfIDMI(Mark, Mark2, self.bdCellyDown, self.Deltay)
        
        #############################################
        # Up
        Mark = np.array([-1, 1])
        Mark2 = np.array([- self.Ny * self.Nz , 0], dtype = np.float64)
        if self.Nx in (1, ): 
            ValueUpx, IUpx, JUpx = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueUpx, IUpx, JUpx = self.PrepareForMatrixOfIDMI(Mark, Mark2, self.bdCellxUp, self.Deltax)
        Mark2 = np.array([- self.Nz , 0], dtype = np.float64)
        if self.Ny in (1, ): 
            ValueUpy, IUpy, JUpy = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
        else:
            ValueUpy, IUpy, JUpy = self.PrepareForMatrixOfIDMI(Mark, Mark2, self.bdCellyUp, self.Deltay)


        Valuex = np.concatenate((ValueInterx.reshape(-1), ValueDownx.reshape(-1), ValueUpx.reshape(-1)))
        Valuey = np.concatenate((ValueIntery.reshape(-1), ValueDowny.reshape(-1), ValueUpy.reshape(-1)))
        Ix = np.concatenate((IInterx.reshape(-1), IDownx.reshape(-1), IUpx.reshape(-1)))
        Iy = np.concatenate((IIntery.reshape(-1), IDowny.reshape(-1), IUpy.reshape(-1)))
        Jx = np.concatenate((JInterx.reshape(-1), JDownx.reshape(-1), JUpx.reshape(-1)))
        Jy = np.concatenate((JIntery.reshape(-1), JDowny.reshape(-1), JUpy.reshape(-1)))
        Value = np.concatenate((Valuex.reshape(-1), Valuey.reshape(-1), - Valuex.reshape(-1), - Valuey.reshape(-1)))
        I = np.concatenate((Ix.reshape(-1), Iy.reshape(-1) + self.NC, Ix.reshape(-1) + 2*self.NC, Iy.reshape(-1) + 2*self.NC))
        J = np.concatenate((Jx.reshape(-1) + 2*self.NC, Jy.reshape(-1) + 2*self.NC, Jx.reshape(-1), Jy.reshape(-1) + self.NC))  
        self.FullMatrixOfIDMI = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(3*self.NC, 3*self.NC))

        return True



    # def MatrixOfDelta(self):
    #     Mark = np.array([1, -2, 1])
    #     Mark2 = np.array([- self.Ny * self.Nz, 0, self.Ny * self.Nz ], dtype = np.float64)
    #     if self.Nx in (1, 2): 
    #         ValueInterx, IInterx, JInterx = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
    #     else:
    #         ValueInterx, IInterx, JInterx = self.PrepareForMatrixOfDelta(Mark, Mark2, self.InvbdCellx, self.Deltax)
    #     Mark2 = np.array([-  self.Nz , 0,  self.Nz ], dtype = np.float64)
    #     if self.Ny in (1, 2): 
    #         ValueIntery, IIntery, JIntery = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
    #     else:
    #         ValueIntery, IIntery, JIntery = self.PrepareForMatrixOfDelta(Mark, Mark2, self.InvbdCelly, self.Deltay)
    #     Mark2 = np.array([- 1, 0, 1], dtype = np.float64)
    #     if self.Nz in (1, 2): 
    #         ValueInterz, IInterz, JInterz = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
    #     else:
    #         ValueInterz, IInterz, JInterz = self.PrepareForMatrixOfDelta(Mark, Mark2, self.InvbdCellz, self.Deltaz)

    #     Mark = np.array([-25/23, 26/23, -1/23])
    #     Mark2 = np.array([0,  self.Ny * self.Nz , 2* self.Ny * self.Nz ], dtype = np.float64)
    #     if self.Nx in (1, 2): 
    #         ValueDownx, IDownx, JDownx = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
    #     else:
    #         ValueDownx, IDownx, JDownx = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellxDown, self.Deltax)
    #     Mark2 = np.array([0,  self.Nz , 2* self.Nz ], dtype = np.float64)
    #     if self.Ny in (1, 2): 
    #         ValueDowny, IDowny, JDowny = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
    #     else:
    #         ValueDowny, IDowny, JDowny = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellyDown, self.Deltay)
    #     Mark2 = np.array([0, 1, 2], dtype = np.float64)
    #     if self.Nz in (1, 2): 
    #         ValueDownz, IDownz, JDownz = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
    #     else:
    #         ValueDownz, IDownz, JDownz = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellzDown, self.Deltaz)

    #     Mark = np.array([-1/23, 26/23, -25/23])
    #     Mark2 = np.array([-2* self.Ny * self.Nz , - self.Ny * self.Nz , 0], dtype = np.float64)
    #     if self.Nx in (1, 2): 
    #         ValueUpx, IUpx, JUpx = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
    #     else:
    #         ValueUpx, IUpx, JUpx = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellxUp, self.Deltax)
    #     Mark2 = np.array([-2* self.Nz , - self.Nz , 0], dtype = np.float64)
    #     if self.Ny in (1, 2): 
    #         ValueUpy, IUpy, JUpy = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
    #     else:
    #         ValueUpy, IUpy, JUpy = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellyUp, self.Deltay)
    #     Mark2 = np.array([-2, -1, 0], dtype = np.float64)
    #     if self.Nz in (1, 2): 
    #         ValueUpz, IUpz, JUpz = np.array([], dtype = np.float64), np.array([], dtype = np.float64), np.array([], dtype = np.float64)
    #     else:
    #         ValueUpz, IUpz, JUpz = self.PrepareForMatrixOfDelta(Mark, Mark2, self.bdCellzUp, self.Deltaz)

    #     Value = np.concatenate((ValueInterx.reshape(-1), ValueIntery.reshape(-1), ValueInterz.reshape(-1), \
    #                             ValueDownx.reshape(-1), ValueDowny.reshape(-1), ValueDownz.reshape(-1), \
    #                             ValueUpx.reshape(-1), ValueUpy.reshape(-1), ValueUpz.reshape(-1)))
    #     I = np.concatenate((IInterx.reshape(-1), IIntery.reshape(-1), IInterz.reshape(-1), \
    #                         IDownx.reshape(-1), IDowny.reshape(-1), IDownz.reshape(-1), \
    #                         IUpx.reshape(-1), IUpy.reshape(-1), IUpz.reshape(-1)))
    #     J = np.concatenate((JInterx.reshape(-1), JIntery.reshape(-1), JInterz.reshape(-1), \
    #                         JDownx.reshape(-1), JDowny.reshape(-1), JDownz.reshape(-1), \
    #                         JUpx.reshape(-1), JUpy.reshape(-1), JUpz.reshape(-1)))
    #     self.Matrix_Delta = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(self.NC, self.NC))
    #     Value = np.concatenate((Value.reshape(-1), Value.reshape(-1), Value.reshape(-1)))
    #     I = np.concatenate((I.reshape(-1), I.reshape(-1) + self.NC, I.reshape(-1) + 2*self.NC))
    #     J = np.concatenate((J.reshape(-1), J.reshape(-1) + self.NC, J.reshape(-1) + 2*self.NC))            
    #     self.FullMatrix_Delta = csr_matrix((Value.flat,(I.flat,J.flat)), shape=(3*self.NC, 3*self.NC))
    #     return True
    
    ######################################################################################################
    ######################################################################################################



