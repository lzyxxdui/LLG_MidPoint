# LLG_3D_StrayField_GPU.py
import numpy as np
import time

# 尝试导入 CuPy，失败则禁用 GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(" GPU detected: Using CuPy for stray field acceleration")
except (ImportError, Exception) as e:
    GPU_AVAILABLE = False
    print(f" CuPy not available ({type(e).__name__}): Falling back to CPU")

class StrayField(object):
    # ========== 辅助函数（完全保留原逻辑）==========
    def FunctionOfDemagnetizationTensorLu(self, u, p):
        return np.log((p + u + 1e-25) / (p - u + 1e-25)) / 2

    def FunctionOfDemagnetizationTensorQu(self, u, v, w, p):
        return u * np.arctan(v * w / (u * p + 1e-25))

    def FunctionOfDemagnetizationTensorF022(self, u, v, w, p, Coe):
        Value = (v * (w**2 - u**2) / 2 * self.FunctionOfDemagnetizationTensorLu(v, p) +
                 w * (v**2 - u**2) / 2 * self.FunctionOfDemagnetizationTensorLu(w, p) -
                 v * w * self.FunctionOfDemagnetizationTensorQu(u, v, w, p) +
                 p * (2 * u**2 - v**2 - w**2) / 6)
        return np.einsum('qweijk,ijk->qwe', Value, Coe)

    def FunctionOfDemagnetizationTensorF112(self, u, v, w, p, Coe):
        Value = (u * v * w * self.FunctionOfDemagnetizationTensorLu(w, p) +
                 v * (3 * w**2 - v**2) / 6 * self.FunctionOfDemagnetizationTensorLu(u, p) +
                 u * (3 * w**2 - u**2) / 6 * self.FunctionOfDemagnetizationTensorLu(v, p) -
                 w / 6 * (w * self.FunctionOfDemagnetizationTensorQu(w, u, v, p) +
                          3 * u * self.FunctionOfDemagnetizationTensorQu(u, v, w, p) +
                          3 * v * self.FunctionOfDemagnetizationTensorQu(v, w, u, p)) -
                 p * u * v / 3)
        return np.einsum('qweijk,ijk->qwe', Value, Coe)

    # ========== 核心：张量计算（CPU执行 + 自动GPU传输）==========
    def ComputeDemagnetizationTensor(self):
        # === 惰性初始化GPU标志（关键修复）===
        if not hasattr(self, 'use_gpu'):
            self.use_gpu = GPU_AVAILABLE
        if not hasattr(self, 'gpu_tensors_transferred'):
            self.gpu_tensors_transferred = False
        
        StartTime = time.time()
        
        # --- 原始CPU张量计算（完全保留）---
        Markx, Marky, Markz = np.arange(self.Nx, dtype=np.float64), np.arange(self.Ny, dtype=np.float64), np.arange(self.Nz, dtype=np.float64)
        rx = np.append(-Markx, Markx[-1:0:-1]) * self.Deltax
        ry = np.append(-Marky, Marky[-1:0:-1]) * self.Deltay
        rz = np.append(-Markz, Markz[-1:0:-1]) * self.Deltaz

        rx = np.broadcast_to(rx[:, None, None], shape=(2*self.Nx-1, 2*self.Ny-1, 2*self.Nz-1))
        ry = np.broadcast_to(ry[None, :, None], shape=(2*self.Nx-1, 2*self.Ny-1, 2*self.Nz-1))
        rz = np.broadcast_to(rz[None, None, :], shape=(2*self.Nx-1, 2*self.Ny-1, 2*self.Nz-1))

        Rx = np.array([rx - self.Deltax, rx, rx + self.Deltax], dtype=np.float64).transpose(1,2,3,0)
        Rx = np.broadcast_to(Rx[..., :, None, None], shape=rx.shape + (3,3,3))
        Ry = np.array([ry - self.Deltay, ry, ry + self.Deltay], dtype=np.float64).transpose(1,2,3,0)
        Ry = np.broadcast_to(Ry[..., None, :, None], shape=ry.shape + (3,3,3))
        Rz = np.array([rz - self.Deltaz, rz, rz + self.Deltaz], dtype=np.float64).transpose(1,2,3,0)
        Rz = np.broadcast_to(Rz[..., None, None, :], shape=rz.shape + (3,3,3))

        p = np.sqrt(Rx**2 + Ry**2 + Rz**2)
        Mark = np.array([-1, 2, -1], dtype=np.float64)
        CoeOfF = Mark.reshape(3,1,1) * Mark.reshape(1,3,1) * Mark.reshape(1,1,3)

        self.Nxx = self.FunctionOfDemagnetizationTensorF022(Rx, Ry, Rz, p, CoeOfF)
        self.Nyy = self.FunctionOfDemagnetizationTensorF022(Ry, Rz, Rx, p, CoeOfF)
        self.Nzz = self.FunctionOfDemagnetizationTensorF022(Rz, Rx, Ry, p, CoeOfF)
        self.Nxy = self.FunctionOfDemagnetizationTensorF112(Rx, Ry, Rz, p, CoeOfF)
        self.Nyz = self.FunctionOfDemagnetizationTensorF112(Ry, Rz, Rx, p, CoeOfF)
        self.Nzx = self.FunctionOfDemagnetizationTensorF112(Rz, Rx, Ry, p, CoeOfF)
        self.Nyx = self.Nxy.copy()
        self.Nzy = self.Nyz.copy()
        self.Nxz = self.Nzx.copy()

        # FFT 转换
        self.fftxyz = np.zeros((3, 3, 2*self.Nx-1, 2*self.Ny-1, 2*self.Nz-1), dtype=np.complex128)
        self.fftxyz[0,0,:] = np.fft.fftn(self.Nxx)
        self.fftxyz[1,1,:] = np.fft.fftn(self.Nyy)
        self.fftxyz[2,2,:] = np.fft.fftn(self.Nzz)
        self.fftxyz[0,1,:] = np.fft.fftn(self.Nxy)
        self.fftxyz[0,2,:] = np.fft.fftn(self.Nxz)
        self.fftxyz[1,2,:] = np.fft.fftn(self.Nyz)
        self.fftxyz[1,0,:] = self.fftxyz[0,1,:].copy()
        self.fftxyz[2,1,:] = self.fftxyz[1,2,:].copy()
        self.fftxyz[2,0,:] = self.fftxyz[0,2,:].copy()
        # --- CPU计算结束 ---

        # === 自动传输到GPU（如可用）===
        if self.use_gpu:
            try:
                self._transfer_tensors_to_gpu()
                print(f" FFT tensors transferred to GPU (size: {2*self.Nx-1}x{2*self.Ny-1}x{2*self.Nz-1})")
            except Exception as e:
                print(f" GPU tensor transfer failed ({type(e).__name__}), falling back to CPU")
                self.use_gpu = False

        EndTime = time.time()
        print(f'Time for Stray Field Tensor: {EndTime - StartTime:.2f}s (GPU: {"enabled" if self.use_gpu else "disabled"})')
        return True

    def _transfer_tensors_to_gpu(self):
        """将预计算的FFT张量传输到GPU"""
        self.gpu_fftxyz = cp.zeros((3, 3, 2*self.Nx-1, 2*self.Ny-1, 2*self.Nz-1), dtype=cp.complex128)
        for i in range(3):
            for j in range(3):
                self.gpu_fftxyz[i, j] = cp.asarray(self.fftxyz[i, j])
        self.gpu_tensors_transferred = True

    # ========== 核心：GPU加速的退磁场计算 ==========
    def Computemu0Hs(self, m):
        # === 惰性初始化（关键修复）===
        if not hasattr(self, 'use_gpu'):
            self.use_gpu = GPU_AVAILABLE
        if not hasattr(self, 'gpu_tensors_transferred'):
            self.gpu_tensors_transferred = False
        
        # 自动回退到CPU的条件
        if not self.use_gpu or not getattr(self, 'gpu_tensors_transferred', False):
            return self._Computemu0Hs_cpu(m)
        
        try:
            # 1. 转换到GPU
            m_cpu = m.reshape(3, self.Nx, self.Ny, self.Nz)
            m_gpu = cp.asarray(m_cpu)
            
            # 2. 零填充
            pad_x, pad_y, pad_z = 2*self.Nx-1, 2*self.Ny-1, 2*self.Nz-1
            Linm0 = cp.zeros((pad_x, pad_y, pad_z), dtype=cp.float64)
            Linm1 = cp.zeros_like(Linm0)
            Linm2 = cp.zeros_like(Linm0)
            
            Linm0[0:self.Nx, 0:self.Ny, 0:self.Nz] = m_gpu[0]
            Linm1[0:self.Nx, 0:self.Ny, 0:self.Nz] = m_gpu[1]
            Linm2[0:self.Nx, 0:self.Ny, 0:self.Nz] = m_gpu[2]
            
            # 3. GPU FFT
            FFTm0 = cp.fft.fftn(Linm0)
            FFTm1 = cp.fft.fftn(Linm1)
            FFTm2 = cp.fft.fftn(Linm2)
            
            # 4. 频域卷积
            Hs_gpu = cp.zeros((3, self.Nx, self.Ny, self.Nz), dtype=cp.float64)
            Hs_gpu[0] = cp.fft.ifftn(
                self.gpu_fftxyz[0,0] * FFTm0 + 
                self.gpu_fftxyz[0,1] * FFTm1 + 
                self.gpu_fftxyz[0,2] * FFTm2
            ).real[0:self.Nx, 0:self.Ny, 0:self.Nz]
            
            Hs_gpu[1] = cp.fft.ifftn(
                self.gpu_fftxyz[1,0] * FFTm0 + 
                self.gpu_fftxyz[1,1] * FFTm1 + 
                self.gpu_fftxyz[1,2] * FFTm2
            ).real[0:self.Nx, 0:self.Ny, 0:self.Nz]
            
            Hs_gpu[2] = cp.fft.ifftn(
                self.gpu_fftxyz[2,0] * FFTm0 + 
                self.gpu_fftxyz[2,1] * FFTm1 + 
                self.gpu_fftxyz[2,2] * FFTm2
            ).real[0:self.Nx, 0:self.Ny, 0:self.Nz]
            
            # 5. 转回CPU并应用系数
            Hs_cpu = cp.asnumpy(Hs_gpu).reshape(-1)
            return - (1.0 / (4 * np.pi) / self.VolumeOfUnitCell) * Hs_cpu
            
        except cp.cuda.memory.OutOfMemoryError:
            print(f" GPU OOM (grid: {self.Nx}x{self.Ny}x{self.Nz}), falling back to CPU")
            self.use_gpu = False
            return self._Computemu0Hs_cpu(m)
        except Exception as e:
            print(f" GPU computation error ({type(e).__name__}: {e}), falling back to CPU")
            self.use_gpu = False
            return self._Computemu0Hs_cpu(m)

    def _Computemu0Hs_cpu(self, m):
        """原始CPU实现（安全回退）"""
        Linm0 = np.zeros((2*self.Nx-1, 2*self.Ny-1, 2*self.Nz-1), dtype=np.float64)
        Linm1 = np.zeros_like(Linm0)
        Linm2 = np.zeros_like(Linm0)

        m_reshaped = m.reshape(3, self.Nx, self.Ny, self.Nz)
        Linm0[0:self.Nx, 0:self.Ny, 0:self.Nz] = m_reshaped[0]
        Linm1[0:self.Nx, 0:self.Ny, 0:self.Nz] = m_reshaped[1]
        Linm2[0:self.Nx, 0:self.Ny, 0:self.Nz] = m_reshaped[2]

        FFTm0 = np.fft.fftn(Linm0)
        FFTm1 = np.fft.fftn(Linm1)
        FFTm2 = np.fft.fftn(Linm2)

        Hs = np.zeros((3, self.Nx, self.Ny, self.Nz), dtype=np.float64)
        Hs[0] = np.fft.ifftn(self.fftxyz[0,0] * FFTm0 + self.fftxyz[0,1] * FFTm1 + self.fftxyz[0,2] * FFTm2).real[0:self.Nx, 0:self.Ny, 0:self.Nz]
        Hs[1] = np.fft.ifftn(self.fftxyz[1,0] * FFTm0 + self.fftxyz[1,1] * FFTm1 + self.fftxyz[1,2] * FFTm2).real[0:self.Nx, 0:self.Ny, 0:self.Nz]
        Hs[2] = np.fft.ifftn(self.fftxyz[2,0] * FFTm0 + self.fftxyz[2,1] * FFTm1 + self.fftxyz[2,2] * FFTm2).real[0:self.Nx, 0:self.Ny, 0:self.Nz]
        return - (1.0 / (4 * np.pi) / self.VolumeOfUnitCell) * Hs.reshape(-1)

    # ========== 能量计算（无需修改）==========
    def StrayFieldEnergy(self, m, HStrayField=None):
        if HStrayField is None:
            HStrayField = self.Computemu0Hs(m)
        return - self.Lxyz**3 * self.mu0 * self.Ms**2 * np.dot(HStrayField, m) / 2 * self.VolumeOfUnitCell

    # ========== 保留原有多进程方法（兼容性）==========
    def Computemu0HsThreeProcess(self, m):
        return self._Computemu0Hs_cpu(m)