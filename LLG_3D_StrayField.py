import numpy as np
from multiprocessing import Pool
import time


class StrayField(object):

    def FunctionOfDemagnetizationTensorLu(self, u, p):
        return np.log((p+u + 1e-25)/(p-u + 1e-25))/2

    def FunctionOfDemagnetizationTensorQu(self, u, v, w, p):
        return u*np.arctan(v*w/(u*p + 1e-25))

    def FunctionOfDemagnetizationTensorF022(self, u, v, w, p, Coe):
        Value = v*(w**2 - u**2)/2* self.FunctionOfDemagnetizationTensorLu(v, p) \
                + w*(v**2 - u**2)/2* self.FunctionOfDemagnetizationTensorLu(w, p) \
                - v*w* self.FunctionOfDemagnetizationTensorQu(u, v, w, p) \
                + p*(2*u**2 - v**2 - w**2)/6 # shape = u.shape
        return np.einsum('qweijk, ijk->qwe', Value, Coe) # (Nx, Ny, Nz)


    def FunctionOfDemagnetizationTensorF112(self, u, v, w, p, Coe):
        Value = u*v*w* self.FunctionOfDemagnetizationTensorLu(w, p) \
                + v* (3*w**2 - v**2)/6 * self.FunctionOfDemagnetizationTensorLu(u, p) \
                + u*(3*w**2 - u**2)/6 * self.FunctionOfDemagnetizationTensorLu(v, p) \
                - w/6 * (w* self.FunctionOfDemagnetizationTensorQu(w, u, v, p) \
                        + 3 * u * self.FunctionOfDemagnetizationTensorQu(u, v, w, p) \
                            + 3 * v * self.FunctionOfDemagnetizationTensorQu(v, w, u, p) ) \
                - p * u * v /3 # (Nx, Ny, Nz, 3, 3, 3)
        return np.einsum('qweijk, ijk->qwe', Value, Coe) # (Nx, Ny, Nz)

    def ComputeDemagnetizationTensor(self):
        StartTime = time.time()
        Markx, Marky, Markz = np.arange(self.Nx, dtype = np.float64), np.arange(self.Ny, dtype = np.float64), np.arange(self.Nz, dtype = np.float64)
        rx = np.append(- Markx, Markx[-1:0:-1]) * self.Deltax # (2Nz -1,)
        ry = np.append(- Marky, Marky[-1:0:-1]) * self.Deltay # (2Ny -1,)
        rz = np.append(- Markz, Markz[-1:0:-1]) * self.Deltaz # (2Nz -1,)


        rx = np.broadcast_to(rx[:, None, None], shape = (2*self.Nx -1, 2*self.Ny -1, 2*self.Nz -1))
        ry = np.broadcast_to(ry[None, :, None], shape = (2*self.Nx -1, 2*self.Ny -1, 2*self.Nz -1))
        rz = np.broadcast_to(rz[None, None, :], shape = (2*self.Nx -1, 2*self.Ny -1, 2*self.Nz -1))   


        Rx = np.array([rx - self.Deltax, rx, rx + self.Deltax], dtype = np.float64).transpose(1, 2, 3, 0) # (3, Nx, Ny, Nz) -> (Nx, Ny, Nz, 3)
        Rx = np.broadcast_to(Rx[..., :, None, None], shape = rx.shape + (3, 3, 3)) # (Nx, Ny, Nz, 3, 3, 3)

        Ry = np.array([ry - self.Deltay, ry, ry + self.Deltay], dtype = np.float64).transpose(1, 2, 3, 0) # (3, Nx, Ny, Nz) -> (Nx, Ny, Nz, 3)
        Ry = np.broadcast_to(Ry[..., None, :, None], shape = ry.shape + (3, 3, 3)) # (Nx, Ny, Nz, 3, 3, 3)

        Rz = np.array([rz - self.Deltaz, rz, rz + self.Deltaz], dtype = np.float64).transpose(1, 2, 3, 0) # (3, Nx, Ny, Nz) -> (Nx, Ny, Nz, 3)
        Rz = np.broadcast_to(Rz[..., None, None, :], shape = rz.shape + (3, 3, 3)) # (Nx, Ny, Nz, 3, 3, 3)

        p = np.sqrt(Rx**2 + Ry**2 + Rz**2)

        Mark = np.array([-1, 2, -1], dtype = np.float64)
        CoeOfF = Mark.reshape(3, 1, 1) * Mark.reshape(1, 3, 1) * Mark.reshape(1, 1, 3) # (3, 3, 3)


        self.Nxx = self.FunctionOfDemagnetizationTensorF022(Rx, Ry, Rz, p, CoeOfF)  # (Nx, Ny, Nz)
        self.Nyy = self.FunctionOfDemagnetizationTensorF022(Ry, Rz, Rx, p, CoeOfF)
        self.Nzz = self.FunctionOfDemagnetizationTensorF022(Rz, Rx, Ry, p, CoeOfF)

        self.Nxy = self.FunctionOfDemagnetizationTensorF112(Rx, Ry, Rz, p, CoeOfF)
        self.Nyz = self.FunctionOfDemagnetizationTensorF112(Ry, Rz, Rx, p, CoeOfF)
        self.Nzx = self.FunctionOfDemagnetizationTensorF112(Rz, Rx, Ry, p, CoeOfF)


        self.Nyx = self.Nxy.copy()
        self.Nzy = self.Nyz.copy()
        self.Nxz = self.Nzx.copy()
        
        self.fftxyz = np.zeros((3, 3, 2*self.Nx - 1, 2*self.Ny - 1, 2*self.Nz - 1), dtype = np.complex128)
        self.fftxyz[0, 0, :] = np.fft.fftn(self.Nxx) # (2*Nx*Ny*Nz -2)
        self.fftxyz[1, 1, :] = np.fft.fftn(self.Nyy) # (2*Nx*Ny*Nz -2)
        self.fftxyz[2, 2, :] = np.fft.fftn(self.Nzz) # (2*Nx*Ny*Nz -2)
        self.fftxyz[0, 1, :] = np.fft.fftn(self.Nxy) # (2*Nx*Ny*Nz -2)
        self.fftxyz[0, 2, :] = np.fft.fftn(self.Nxz) # (2*Nx*Ny*Nz -2)
        self.fftxyz[1, 2, :] = np.fft.fftn(self.Nyz) # (2*Nx*Ny*Nz -2)  
        self.fftxyz[1, 0, :] = self.fftxyz[0, 1, :].copy()
        self.fftxyz[2, 1, :] = self.fftxyz[1, 2, :].copy()
        self.fftxyz[2, 0, :] = self.fftxyz[0, 2, :].copy()

        EndTime = time.time()
        print('Time for Stray Field Tensor', EndTime - StartTime)
        return True
         
    def Computemu0Hs(self, m):
        Linm0 = np.zeros((2*self.Nx -1, 2*self.Ny -1, 2*self.Nz -1), dtype = np.float64)
        Linm1 = np.zeros((2*self.Nx -1, 2*self.Ny -1, 2*self.Nz -1), dtype = np.float64)
        Linm2 = np.zeros((2*self.Nx -1, 2*self.Ny -1, 2*self.Nz -1), dtype = np.float64)

        Linm0[0:self.Nx, 0:self.Ny, 0:self.Nz] = m.reshape(3, self.Nx, self.Ny, self.Nz)[0, :]
        Linm1[0:self.Nx, 0:self.Ny, 0:self.Nz] = m.reshape(3, self.Nx, self.Ny, self.Nz)[1, :]
        Linm2[0:self.Nx, 0:self.Ny, 0:self.Nz] = m.reshape(3, self.Nx, self.Ny, self.Nz)[2, :]

        FFTm0 = np.fft.fftn(Linm0)
        FFTm1 = np.fft.fftn(Linm1)
        FFTm2 = np.fft.fftn(Linm2)

        Hs = np.zeros((3, self.Nx, self.Ny, self.Nz), dtype = np.float64)
        Hs[0, :] = np.fft.ifftn(self.fftxyz[0, 0, :] * FFTm0 + self.fftxyz[0, 1, :] * FFTm1 + self.fftxyz[0, 2, :] * FFTm2).real[0:self.Nx, 0:self.Ny, 0:self.Nz]
        Hs[1, :] = np.fft.ifftn(self.fftxyz[1, 0, :] * FFTm0 + self.fftxyz[1, 1, :] * FFTm1 + self.fftxyz[1, 2, :] * FFTm2).real[0:self.Nx, 0:self.Ny, 0:self.Nz]
        Hs[2, :] = np.fft.ifftn(self.fftxyz[2, 0, :] * FFTm0 + self.fftxyz[2, 1, :] * FFTm1 + self.fftxyz[2, 2, :] * FFTm2).real[0:self.Nx, 0:self.Ny, 0:self.Nz]
        return - (1/(4*np.pi)/self.VolumeOfUnitCell) * Hs.reshape(-1)

#     def HandleForComputemu0HsThreeProcess0(self, args):
#         Linm, HsType = args
#         StartTime = time.time()
#         Linm0 = np.zeros((2*self.Nx -1, 2*self.Ny -1, 2*self.Nz -1), dtype = np.float64)
#         Linm0[0:self.Nx, 0:self.Ny, 0:self.Nz] = Linm
#         FFTm0 = np.fft.fftn(Linm0)
#         LinAA = np.zeros((3, 2*self.Nx -1, 2*self.Ny -1, 2*self.Nz -1), dtype = np.complex64)
#         for i in range(3): LinAA = self.fftxyz[i, HsType, :] * FFTm0
#         EndTime = time.time()
#         print('Time in Handle0', HsType, EndTime - StartTime)
#         return LinAA

#     def HandleForComputemu0HsThreeProcess1(self, FFTXYZFFTM):
#         return np.fft.ifftn(FFTXYZFFTM).real[0:self.Nx, 0:self.Ny, 0:self.Nz]

#     def Computemu0HsThreeProcess(self, m):
#         Linm = m.reshape(3, self.Nx, self.Ny, self.Nz)
#         pool = Pool(processes=2)  # 创建一个包含两个进程的进程池

#         StartTime0 = time.time()
#         results0 = pool.map(self.HandleForComputemu0HsThreeProcess0, [(Linm[0, :], 0), (Linm[1, :], 1)])

#         MiddleTime0 = time.time()

#         FFTXYZFFTM0 = np.zeros((3, 3, 2*self.Nx -1, 2*self.Ny -1, 2*self.Nz -1), dtype = np.complex64)

#         MiddleTime1 = time.time()


#         FFTXYZFFTM0[:, 2, :] = self.HandleForComputemu0HsThreeProcess0((Linm[2, :], 2))


#         MiddleTime2 = time.time()

#         FFTXYZFFTM0[:, 0, :] = results0[0]
#         FFTXYZFFTM0[:, 1, :] = results0[1]

#         EndTime0 = time.time()
#         print('Time 0', MiddleTime0 - StartTime0, MiddleTime1 - MiddleTime0, MiddleTime2 - MiddleTime1, EndTime0 - MiddleTime2)


#         StartTime0 = time.time()
#         results1 = pool.map(self.HandleForComputemu0HsThreeProcess1, [FFTXYZFFTM0[0, 0, :] + FFTXYZFFTM0[0, 1, :] + FFTXYZFFTM0[0, 2, :], FFTXYZFFTM0[1, 0, :] + FFTXYZFFTM0[1, 1, :] + FFTXYZFFTM0[1, 2, :]])
#         pool.close()

#         Hs = np.zeros((3, self.Nx, self.Ny, self.Nz), dtype = np.float64)
#         Hs[2, :] = self.HandleForComputemu0HsThreeProcess1(FFTXYZFFTM0[2, 0, :] + FFTXYZFFTM0[2, 1, :] + FFTXYZFFTM0[2, 2, :])
#         Hs[0, :] = results1[0]
#         Hs[1, :] = results1[1]
#         pool.join()
#         EndTime0 = time.time()
#         print('Time 1', EndTime0 - StartTime0)
#         return - (self.mu0/(4*np.pi)/self.VolumeOfUnitCell) * Hs.reshape(-1)

    def StrayFieldEnergy(self, m, HStrayField = None):
        if HStrayField is None: HStrayField = self.Computemu0Hs(m)
        return - self.Lxyz**3 * self.mu0 * self.Ms**2 * np.dot(HStrayField, m) /2 * self.VolumeOfUnitCell
