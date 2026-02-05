import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# if self.FigureName is not None: plt.savefig(self.FigureName + '_ChangeOfEnergy.eps')

class FigureAndSaveData(object):   
    
    def SaveDataAndFig(self, m, j):
        if self.FileName is not None: np.save(self.DataName + "_%s" %j, m)
        return True

    def FigureForFace(self, m, FaceType = "xy", Position = 0, Scale = (1, 1), ReNorm = True, \
        Color = None, MathTextType = False):
        if Position == 'Bottom': Position = 0
        if Position == 'Top': Position = -1
        if FaceType == "xz":
            Lin1 = self.SpaceDomain[0] + np.arange(self.Nx, dtype = np.float64)*self.Deltax + self.Deltax/2
            Lin2 = self.SpaceDomain[4] + np.arange(self.Nz, dtype = np.float64)*self.Deltaz + self.Deltaz/2
            if Position == 'Middle': Position = int(self.Ny/2) + 1
            u = m.reshape(3, self.Nx, self.Ny, self.Nz)[0, :, Position, :] 
            v = m.reshape(3, self.Nx, self.Ny, self.Nz)[2, :, Position, :] 
            if Color == "x":
                w = u.copy()
            elif Color == "z":
                w = v.copy()
            elif Color == "y":
                w = m.reshape(3, self.Nx, self.Ny, self.Nz)[1, :, Position, :]
            XYExtent = np.array([self.SpaceDomain[0], self.SpaceDomain[1], self.SpaceDomain[4], self.SpaceDomain[5]])
        elif FaceType == "xy":
            Lin1 = self.SpaceDomain[0] + np.arange(self.Nx, dtype = np.float64)*self.Deltax + self.Deltax/2
            Lin2 = self.SpaceDomain[2] + np.arange(self.Ny, dtype = np.float64)*self.Deltay + self.Deltay/2
            if Position == 'Middle': Position = int(self.Nz/2) + 1
            u = m.reshape(3, self.Nx, self.Ny, self.Nz)[0, :, :, Position] 
            v = m.reshape(3, self.Nx, self.Ny, self.Nz)[1, :, :, Position] 
            if Color == "x":
                w = u.copy()
            elif Color == "y":
                w = v.copy()
            elif Color == "z":
                w = m.reshape(3, self.Nx, self.Ny, self.Nz)[2, :, :, Position]
            XYExtent = np.array([self.SpaceDomain[0], self.SpaceDomain[1], self.SpaceDomain[2], self.SpaceDomain[3]])
        elif FaceType == "yz":
            Lin1 = self.SpaceDomain[2] + np.arange(self.Ny, dtype = np.float64)*self.Deltay + self.Deltay/2
            Lin2 = self.SpaceDomain[4] + np.arange(self.Nz, dtype = np.float64)*self.Deltaz + self.Deltaz/2
            if Position == 'Middle': Position = int(self.Nx/2) + 1
            u = m.reshape(3, self.Nx, self.Ny, self.Nz)[1, Position, :, :]
            v = m.reshape(3, self.Nx, self.Ny, self.Nz)[2, Position, :, :]
            if Color == "y":
                w = u.copy()
            elif Color == "z":
                w = v.copy()
            elif Color == "x":
                w = m.reshape(3, self.Nx, self.Ny, self.Nz)[0, Position, :, :]
            XYExtent = np.array([self.SpaceDomain[2], self.SpaceDomain[3], self.SpaceDomain[4], self.SpaceDomain[5]])
        else:
            print("Please input right FaceType!")
            return False

        if Lin1.shape[0] % Scale[0] ==0 and Lin2.shape[0]%Scale[1] ==0:
            ReduceLin1 = int(Lin1.shape[0]/Scale[0])
            ReduceLin2 = int(Lin2.shape[0]/Scale[1])
        else:
            print("Plear input right Scale!")
            return False

        Lin1 = np.mean(Lin1.reshape(ReduceLin1, Scale[0]), axis = -1)
        Lin2 = np.mean(Lin2.reshape(ReduceLin2, Scale[1]), axis = -1)
        u = u.reshape(ReduceLin1, Scale[0], ReduceLin2, Scale[1]).transpose(0, 2, 1, 3)\
            .reshape(ReduceLin1, ReduceLin2, Scale[0]*Scale[1])
        u = np.mean(u, axis = -1)
        v = v.reshape(ReduceLin1, Scale[0], ReduceLin2, Scale[1]).transpose(0, 2, 1, 3)\
            .reshape(ReduceLin1, ReduceLin2, Scale[0]*Scale[1])
        v = np.mean(v, axis = -1)


        Figurex = np.broadcast_to(Lin1[:, None], shape = Lin1.shape + Lin2.shape)
        Figurey = np.broadcast_to(Lin2[None, :], shape = Lin1.shape + Lin2.shape)
        
        if ReNorm == True:
            Norm = np.sqrt(u**2 + v**2) + 1e-20
            u = u/Norm
            v = v/Norm
        
        fig, ax = plt.subplots()
        plt.rcParams['figure.figsize'] = (8*int((XYExtent[1]-XYExtent[0])/min(XYExtent[1]-XYExtent[0], XYExtent[3]-XYExtent[2])), 8*int((XYExtent[3]-XYExtent[2])/min(XYExtent[1]-XYExtent[0], XYExtent[3]-XYExtent[2])))
        ax.set_aspect('equal')
        q = ax.quiver(Figurex, Figurey, u, v)
        if Color is not None:
            # print('xlim', ax.get_xlim(), ax.get_ylim())
            # plt.imshow(np.swapaxes(w, 0, 1), cmap = 'coolwarm', vmin=-1, vmax=1, extent = [0, 128-9, 0, 128-9])
            print('w', w.shape)
            print('w', w)
            # plt.imshow(np.swapaxes(w, 0, 1), cmap = 'coolwarm', vmin=-1, vmax=1, extent = XYExtent)
            plt.imshow(np.swapaxes(w[:, -1::-1], 0, 1), cmap = 'coolwarm', vmin=-1, vmax=1, extent = XYExtent)
            # plt.colorbar()
            # print('xlim', ax.get_xlim(), ax.get_ylim())
        # 将useMathText设置为True,使得刻度标记显示为科学计数法
        if MathTextType == True:
            from matplotlib.ticker import ScalarFormatter
            y_formatter = ScalarFormatter(useMathText=True)
            # 控制刻度标记的科学计数法显示
            y_formatter.set_powerlimits((-2, 2))  
            ax.yaxis.set_major_formatter(y_formatter)
            ax.xaxis.set_major_formatter(y_formatter)
        plt.show()
        return True
    

    def FigureForFaceArrowColor(self, m, FaceType = "xy", Position = 0, Scale = (1, 1), ReNorm = True, \
        Color = None, MathTextType = False):
        if Position == 'Bottom': Position = 0
        if Position == 'Top': Position = -1
        if FaceType == "xz":
            Lin1 = self.SpaceDomain[0] + np.arange(self.Nx, dtype = np.float64)*self.Deltax + self.Deltax/2
            Lin2 = self.SpaceDomain[4] + np.arange(self.Nz, dtype = np.float64)*self.Deltaz + self.Deltaz/2
            if Position == 'Middle': Position = int(self.Ny/2) + 1
            u = m.reshape(3, self.Nx, self.Ny, self.Nz)[0, :, Position, :] 
            v = m.reshape(3, self.Nx, self.Ny, self.Nz)[2, :, Position, :] 
            if Color == "x":
                w = u.copy()
            elif Color == "z":
                w = v.copy()
            elif Color == "y":
                w = m.reshape(3, self.Nx, self.Ny, self.Nz)[1, :, Position, :]
            XYExtent = np.array([self.SpaceDomain[0], self.SpaceDomain[1], self.SpaceDomain[4], self.SpaceDomain[5]])
        elif FaceType == "xy":
            Lin1 = self.SpaceDomain[0] + np.arange(self.Nx, dtype = np.float64)*self.Deltax + self.Deltax/2
            Lin2 = self.SpaceDomain[2] + np.arange(self.Ny, dtype = np.float64)*self.Deltay + self.Deltay/2
            if Position == 'Middle': Position = int(self.Nz/2) + 1
            u = m.reshape(3, self.Nx, self.Ny, self.Nz)[0, :, :, Position] 
            v = m.reshape(3, self.Nx, self.Ny, self.Nz)[1, :, :, Position] 
            if Color == "x":
                w = u.copy()
            elif Color == "y":
                w = v.copy()
            elif Color == "z":
                w = m.reshape(3, self.Nx, self.Ny, self.Nz)[2, :, :, Position]
            XYExtent = np.array([self.SpaceDomain[0], self.SpaceDomain[1], self.SpaceDomain[2], self.SpaceDomain[3]])
        elif FaceType == "yz":
            Lin1 = self.SpaceDomain[2] + np.arange(self.Ny, dtype = np.float64)*self.Deltay + self.Deltay/2
            Lin2 = self.SpaceDomain[4] + np.arange(self.Nz, dtype = np.float64)*self.Deltaz + self.Deltaz/2
            if Position == 'Middle': Position = int(self.Nx/2) + 1
            u = m.reshape(3, self.Nx, self.Ny, self.Nz)[1, Position, :, :]
            v = m.reshape(3, self.Nx, self.Ny, self.Nz)[2, Position, :, :]
            if Color == "y":
                w = u.copy()
            elif Color == "z":
                w = v.copy()
            elif Color == "x":
                w = m.reshape(3, self.Nx, self.Ny, self.Nz)[0, Position, :, :]
            XYExtent = np.array([self.SpaceDomain[2], self.SpaceDomain[3], self.SpaceDomain[4], self.SpaceDomain[5]])
        else:
            print("Please input right FaceType!")
            return False

        if Lin1.shape[0] % Scale[0] ==0 and Lin2.shape[0]%Scale[1] ==0:
            ReduceLin1 = int(Lin1.shape[0]/Scale[0])
            ReduceLin2 = int(Lin2.shape[0]/Scale[1])
        else:
            print("Plear input right Scale!")
            return False

        Lin1 = np.mean(Lin1.reshape(ReduceLin1, Scale[0]), axis = -1)
        Lin2 = np.mean(Lin2.reshape(ReduceLin2, Scale[1]), axis = -1)
        u = u.reshape(ReduceLin1, Scale[0], ReduceLin2, Scale[1]).transpose(0, 2, 1, 3)\
            .reshape(ReduceLin1, ReduceLin2, Scale[0]*Scale[1])
        u = np.mean(u, axis = -1)
        v = v.reshape(ReduceLin1, Scale[0], ReduceLin2, Scale[1]).transpose(0, 2, 1, 3)\
            .reshape(ReduceLin1, ReduceLin2, Scale[0]*Scale[1])
        v = np.mean(v, axis = -1)
        w = w.reshape(ReduceLin1, Scale[0], ReduceLin2, Scale[1]).transpose(0, 2, 1, 3)\
            .reshape(ReduceLin1, ReduceLin2, Scale[0]*Scale[1])
        w = np.mean(w, axis = -1)

        Figurex = np.broadcast_to(Lin1[:, None], shape = Lin1.shape + Lin2.shape)
        Figurey = np.broadcast_to(Lin2[None, :], shape = Lin1.shape + Lin2.shape)
        
        if ReNorm == True:
            Norm = np.sqrt(u**2 + v**2) + 1e-20
            u = u/Norm
            v = v/Norm
        
        fig, ax = plt.subplots()
        plt.rcParams['figure.figsize'] = (8*int((XYExtent[1]-XYExtent[0])/min(XYExtent[1]-XYExtent[0], XYExtent[3]-XYExtent[2])), 8*int((XYExtent[3]-XYExtent[2])/min(XYExtent[1]-XYExtent[0], XYExtent[3]-XYExtent[2])))
        ax.set_aspect('equal')
        
        if Color is not None:
            # q = ax.quiver(Figurex, Figurey, u, v, np.swapaxes(w, 0, 1), cmap = 'coolwarm')
            q = ax.quiver(Figurex, Figurey, u, v, w, cmap = 'coolwarm')
            # plt.imshow(np.swapaxes(w, 0, 1), cmap = 'coolwarm', vmin=-1, vmax=1, extent = XYExtent)
            plt.colorbar(q)
        else:
            q = ax.quiver(Figurex, Figurey, u, v)
        # 将useMathText设置为True,使得刻度标记显示为科学计数法
        if MathTextType == True:
            from matplotlib.ticker import ScalarFormatter
            y_formatter = ScalarFormatter(useMathText=True)
            # 控制刻度标记的科学计数法显示
            y_formatter.set_powerlimits((-2, 2))  
            ax.yaxis.set_major_formatter(y_formatter)
            ax.xaxis.set_major_formatter(y_formatter)
        plt.show()
        return True




    def FigureForVectorFieldanimate(self, m, Type = "xy", Position = 0, Scale = (1, 1), ReNorm = True, Color = None, magnitude = 'full', gif = False):
        plt.rcParams['figure.figsize'] = (12, 12)
        if Position == 'Bottom': Position = 0
        if Position == 'Top': Position = -1
        if Type == "xz":
            Lin1 = self.SpaceDomain[0] + np.arange(self.Nx, dtype = np.float64)*self.Deltax + self.Deltax/2
            Lin2 = self.SpaceDomain[4] + np.arange(self.Nz, dtype = np.float64)*self.Deltaz + self.Deltaz/2
            if Position == 'Middle': Position = int(self.Ny/2) + 1
            u = m.reshape(-1, 3, self.Nx, self.Ny, self.Nz)[:, 0, :, Position, :] 
            v = m.reshape(-1, 3, self.Nx, self.Ny, self.Nz)[:, 2, :, Position, :] 
            if Color == "x":
                w = u.copy()
            elif Color == "z":
                w = v.copy()
            elif Color == "y":
                w = m.reshape(-1, 3, self.Nx, self.Ny, self.Nz)[:, 1, :, Position, :] 
        elif Type == "xy":
            Lin1 = self.SpaceDomain[0] + np.arange(self.Nx, dtype = np.float64)*self.Deltax + self.Deltax/2
            Lin2 = self.SpaceDomain[2] + np.arange(self.Ny, dtype = np.float64)*self.Deltay + self.Deltay/2
            if Position == 'Middle': Position = int(self.Nz/2) + 1
            u = m.reshape(-1, 3, self.Nx, self.Ny, self.Nz)[:, 0, :, :, Position] 
            v = m.reshape(-1, 3, self.Nx, self.Ny, self.Nz)[:, 1, :, :, Position] 
            if Color == "x":
                w = u.copy()
            elif Color == "y":
                w = v.copy()
            elif Color == "z":
                w = m.reshape(-1, 3, self.Nx, self.Ny, self.Nz)[:, 2, :, :, Position] 
        elif Type == "yz":
            Lin1 = self.SpaceDomain[2] + np.arange(self.Ny, dtype = np.float64)*self.Deltay + self.Deltay/2
            Lin2 = self.SpaceDomain[4] + np.arange(self.Nz, dtype = np.float64)*self.Deltaz + self.Deltaz/2
            if Position == 'Middle': Position = int(self.Nx/2) + 1
            u = m.reshape(-1, 3, self.Nx, self.Ny, self.Nz)[:, 1, Position, :, :]
            v = m.reshape(-1, 3, self.Nx, self.Ny, self.Nz)[:, 2, Position, :, :]
            if Color == "y":
                w = u.copy()
            elif Color == "z":
                w = v.copy()
            elif Color == "x":
                w = m.reshape(-1, 3, self.Nx, self.Ny, self.Nz)[:, 0, Position, :, :]
        else:
            print("Please input right Type!")
            return False

        if Lin1.shape[0] % Scale[0] ==0 and Lin2.shape[0]%Scale[1] ==0:
            ReduceLin1 = int(Lin1.shape[0]/Scale[0])
            ReduceLin2 = int(Lin2.shape[0]/Scale[1])
        else:
            print("Plear input right Scale!")
            return False

        Lin1 = np.mean(Lin1.reshape(ReduceLin1, Scale[0]), axis = -1)
        Lin2 = np.mean(Lin2.reshape(ReduceLin2, Scale[1]), axis = -1)
        u = u.reshape(-1, ReduceLin1, Scale[0], ReduceLin2, Scale[1]).transpose(0, 1, 3, 2, 4)\
            .reshape(-1, ReduceLin1, ReduceLin2, Scale[0]*Scale[1])
        u = np.mean(u, axis = -1)
        v = v.reshape(-1, ReduceLin1, Scale[0], ReduceLin2, Scale[1]).transpose(0, 1, 3, 2, 4)\
            .reshape(-1, ReduceLin1, ReduceLin2, Scale[0]*Scale[1])
        v = np.mean(v, axis = -1)


        Figurex = np.broadcast_to(Lin1[:, None], shape = Lin1.shape + Lin2.shape)
        Figurey = np.broadcast_to(Lin2[None, :], shape = Lin1.shape + Lin2.shape)
        
        if ReNorm == True:
            Norm = np.sqrt(u**2 + v**2) + 1e-20
            u = u/Norm
            v = v/Norm
            
            

        fig, ax = plt.subplots()
        
        # print('u', u)
        # print('v', v)    
        # print('shape', u.shape, v.shape, Figurex.shape, Figurey.shape)
        # q = ax.quiver(Figurex, Figurey, u, v)
        # # 保存当前坐标轴状态
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        
        # print('xlim', ax.get_xlim(), ax.get_ylim())
        
        
        if m.ndim == 1:
            # w = w.reshape(ReduceLin1, Scale[0], ReduceLin2, Scale[1]).transpose(0, 2, 1, 3)\
            #     .reshape(ReduceLin1, ReduceLin2, Scale[0]*Scale[1])
            # w = np.mean(w, axis = -1)
            # print('shape', u.shape, v.shape, w.shape)
            # print('xlim', ax.get_xlim(), ax.get_ylim())
            if magnitude == 'full':
                plt.imshow(np.swapaxes(w[0, :], 0, 1), cmap = 'coolwarm', vmin=-self.Ms, vmax=self.Ms)
            if magnitude == 'unit':
                plt.imshow(np.swapaxes(w[0, :], 0, 1), cmap = 'coolwarm', vmin=-1, vmax=1)
            # print('xlim', ax.get_xlim(), ax.get_ylim())
            plt.colorbar()

        else:
        #画动图功能：
            def animate(i):
                plt.cla()  # clear the current plot
                if magnitude == 'full':
                    # plt.imshow(np.swapaxes(w[i, :], 0, 1), cmap = 'coolwarm', vmin=-self.Ms, vmax=self.Ms)
                    plt.imshow(np.swapaxes(w[i, :], 0, 1), cmap = 'coolwarm', vmin=-1, vmax=1)
                if magnitude == 'unit':
                    plt.imshow(np.swapaxes(w[i, :], 0, 1), cmap = 'coolwarm', vmin=-1, vmax=1)
                plt.xlabel('%s' %i)


            # Create the animation
            animation = FuncAnimation(plt.gcf(), animate, frames = np.arange(m.shape[0]), interval=200)
            animation.save("D:/OneDrive/Documents/Gif.gif", writer='pillow', fps=10)
        # ax.set_aspect('equal')
        plt.show()
        return True

    def Figure3D(self, m0, Redu = False, length = 0.000000002):
        mQ = m0.reshape(3, self.Nx, self.Ny, self.Nz)
        # Preparing for Vector Figure
        Linx = self.SpaceDomain[0] + np.arange(self.Nx, dtype = np.float64)*self.Deltax + self.Deltax/2
        Figurex = np.broadcast_to(Linx[:, None, None], shape = (self.Nx, self.Ny, self.Nz))
        Liny = self.SpaceDomain[2] + np.arange(self.Ny, dtype = np.float64)*self.Deltay + self.Deltay/2
        Figurey = np.broadcast_to(Liny[None, :, None], shape = (self.Nx, self.Ny, self.Nz))
        Linz = self.SpaceDomain[4] + np.arange(self.Nz, dtype = np.float64)*self.Deltaz + self.Deltaz/2
        Figurez = np.broadcast_to(Linz[None, None, :], shape = (self.Nx, self.Ny, self.Nz))
        
        u = mQ[0, :]
        v = mQ[1, :]
        w = mQ[2, :]
        Norm = np.sqrt(u**2 + v**2 + w**2) + 1e-20
        u = u/Norm
        v = v/Norm
        w = w/Norm
        
        if Redu == True:
            print('The part of Rede is under doing')
            return False
        
        ax = plt.figure().add_subplot(projection='3d')
        # ax.quiver(Figurex, Figurey, Figurez, mQ[0, :], mQ[1, :], mQ[2, :], length=0.1, normalize=True)
        ax.quiver(Figurex, Figurey, Figurez, u, v, w, length= length)  
        plt.show()
        return True
        
    def FigureEachEnergy(self, Timex, Energy, NumOflinewidth = 0.1, XLABEL = 'Time', YLABEL = 'Each component of energy', InsideLABEL = None, \
        axfontsize = 15, Legend = True, LegendFontsize = 15, markersize = 8, COLOR = ['bo-', 'ro-', 'go-', 'ko-', 'mo-', 'co-'], MathTextType = False):

        plt.figure()
        plt.rcParams['figure.figsize'] = (12, 6)
        ax = plt.gca()  # 获取当前轴（Axes）

        '''
        # 'bo--' 表示蓝色实线, 数据点以实心原点标注
        # 其他的数据点选择: 's' 方块状; 'o' 实心圆; '^'正三角形; 'v'反三角形; '+' 加号; '*'星号; 'x' x号; 'p' 五角星; '1' 和 '2'
        '''

        if Energy.ndim == 1:
            if InsideLABEL is None: InsideLABEL = 'Total Energy'
            plt.plot(Timex, Energy, COLOR[0], alpha = 0.5, linewidth = NumOflinewidth, label = InsideLABEL[0], markersize = markersize)
        elif Energy.ndim == 2:
            if InsideLABEL is None: 
                InsideLABEL = ['Total Energy', 'Anisotropy Energy', 'Exchange Energy', 'Magnetostatic Energy', 'External Energy', 'IDMI Energy']
                NUM = (0,) + tuple(x + 1 for x in self.TypeEffective)
                for i in range(len(self.TypeEffective)+1):
                    plt.plot(Timex, Energy[:, i], COLOR[NUM[i]], alpha = 0.5, linewidth = NumOflinewidth, label = InsideLABEL[NUM[i]], markersize = markersize)
            else:
                for i in range(Energy.shape[-1]):
                    plt.plot(Timex, Energy[:, i], COLOR[i], alpha = 0.5, linewidth = NumOflinewidth, label = InsideLABEL[i], markersize = markersize)

        if Legend: plt.legend(fontsize = LegendFontsize) # 显示上面的 label 
        plt.xlabel(XLABEL, size = 15) # x_label
        plt.ylabel(YLABEL, size = 15) # y_label
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        ax.yaxis.get_offset_text().set_fontsize(axfontsize) # 设置科学记数法的字体大小
        # 将useMathText设置为True,使得刻度标记显示为科学计数法
        if MathTextType == True:
            from matplotlib.ticker import ScalarFormatter
            y_formatter = ScalarFormatter(useMathText=True)
            # 控制刻度标记的科学计数法显示
            y_formatter.set_powerlimits((-2, 2))  
            ax.yaxis.set_major_formatter(y_formatter)
            ax.xaxis.set_major_formatter(y_formatter)
        plt.show()
        # plt.show()
        return True


