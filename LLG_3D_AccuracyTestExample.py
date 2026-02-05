import numpy as np

class AccuracyTestExample(object):
    
    def RunAccuracyTestExample(self,):
        if self.ATExample == 0:
            self.AccuracyTestExample0()
        elif self.ATExample == 1:
            self.AccuracyTestExample1()
        elif self.ATExample == 2:
            self.AccuracyTestExample2()
            
    def AccuracyTestExample0(self):
        def Exact(p, tn):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            me = np.zeros((3,) + x.shape, dtype = np.float64)
            me[0, :] = np.cos(x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2) * np.sin(1/tn)
            me[1, :] = np.sin(x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2) * np.sin(1/tn)
            me[2, :] = np.cos(1/tn) + 0*x
            return me # (3, p.shape[0:-1])

        def Exact_t(p, tn):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]

            met = np.zeros((3,) + x.shape, dtype = np.float64)

            met[0, :] = np.cos(x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2) * np.cos(1/tn) * (-1 /tn**2)
            met[1, :] = np.sin(x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2) * np.cos(1/tn) * (-1 /tn**2)
            met[2, :] = np.sin(1/tn) /tn**2 + 0*x

            return met # (3, p.shape[0:-1])


        def Exact_xx(p, tn):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]

            mexx = np.zeros((3,) + x.shape, dtype = np.float64)

            mexx[0, :] = np.sin(1/tn) * ( \
                        - np.cos(x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2) * ( \
                        (  (2*x - 6*x**2 + 4* x**3) * y**2 * (1-y)**2 * z**2 * (1-z)**2)**2 + ( ( 2*y - 6*y**2 + 4* y**3) * x**2 * (1-x)**2 * z**2 * (1-z)**2)**2 + (( 2*z - 6*z**2 + 4* z**3) * y**2 * (1-y)**2 * x**2 * (1-x)**2)**2 )\
                        - np.sin(x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2) * \
                            ( (2 - 12*x + 12* x**2) * y**2 * (1-y)**2 * z**2 * (1-z)**2 +  (2 - 12*y + 12* y**2) * x**2 * (1-x)**2 * z**2 * (1-z)**2 + (2 - 12*z + 12* z**2) * y**2 * (1-y)**2 * x**2 * (1-x)**2 )
                )
            mexx[1, :] = np.sin(1/tn) * ( \
                        - np.sin(x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2) * ( \
                        (  (2*x - 6*x**2 + 4* x**3) * y**2 * (1-y)**2 * z**2 * (1-z)**2)**2 + ( ( 2*y - 6*y**2 + 4* y**3) * x**2 * (1-x)**2 * z**2 * (1-z)**2)**2 + (( 2*z - 6*z**2 + 4* z**3) * y**2 * (1-y)**2 * x**2 * (1-x)**2)**2 )\
                        + np.cos(x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2) * \
                            ( (2 - 12*x + 12* x**2) * y**2 * (1-y)**2 * z**2 * (1-z)**2 +  (2 - 12*y + 12* y**2) * x**2 * (1-x)**2 * z**2 * (1-z)**2 + (2 - 12*z + 12* z**2) * y**2 * (1-y)**2 * x**2 * (1-x)**2 )
                )
            mexx[2, :] = 0*x
            return mexx # (3, p.shape[0:-1])
        
        self.ExactFun = Exact        
        self.Exact_tFun = Exact_t
        self.Exact_xxFun = Exact_xx
        
        def ForcingForLLG(p, tn):
            me = self.ExactFun(p, tn)
            met = self.Exact_tFun(p, tn)
            mexx = self.Exact_xxFun(p, tn)
            F = met.reshape(-1) + self.OutProduct(me, mexx) - self.alpha * self.OutProduct(me, met)
            if 0 in self.TypeEffective:
                Ani = np.broadcast_to(self.AnisotropyDirection[:, None], shape = me.shape)
                F +=  (self.OutProduct(me, Ani).reshape(3, -1) * np.einsum('qj, qj->j', me, Ani)).reshape(-1)
            return F


        def ForcingForLL(p, tn):
            me = self.ExactFun(p, tn)
            met = self.Exact_tFun(p, tn)
            mexx = self.Exact_xxFun(p, tn)
            meXmexx = self.OutProduct(me, mexx)
            F = met.reshape(-1) + meXmexx + self.alpha * self.OutProduct(me, meXmexx)
            if 0 in self.TypeEffective:
                raise print('This part have not been finished!')
            return F
        
        if self.ATExampleType == "LLG":
            self.ForcingItem = ForcingForLLG
            print("The Example of LLG is set!")
        elif self.ATExampleType == "LL":
            self.ForcingItem = ForcingForLL
            print("The Example of LL is set!")

        self.Ms = 1
        self.mu0 = 1
        self.Cex = 1/2
        if 0 in self.TypeEffective:
            self.TypeEffective = (0, 1,)
        else:
            self.TypeEffective = (1,)
        self.Ku = 1/2
        print('self.TypeEffective is set to', self.TypeEffective)
        print('Ms, mu0, Cex, Ku are set to', self.Ms, self.mu0, self.Cex, self.Ku)
        return True



    def AccuracyTestExample1(self):
        import sympy
        x = sympy.Symbol('x')
        y = sympy.Symbol('y')
        z = sympy.Symbol('z')
        t = sympy.Symbol('t')

        me0 = sympy.cos(x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2 * 1e4) * sympy.sin(t)
        me1 = sympy.sin(x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2 * 1e4) * sympy.sin(t)
        me2 = sympy.cos(t) + 0*x

        mexx0 = sympy.diff(me0, x, 2) + sympy.diff(me0, y, 2) + sympy.diff(me0, z, 2)
        mexx1 = sympy.diff(me1, x, 2) + sympy.diff(me1, y, 2) + sympy.diff(me1, z, 2)
        mexx2 = sympy.diff(me2, x, 2) + sympy.diff(me2, y, 2) + sympy.diff(me2, z, 2)

        met0 = sympy.diff(me0, t)
        met1 = sympy.diff(me1, t)
        met2 = sympy.diff(me2, t)

        Exact0 = sympy.lambdify((x, y, z, t), me0, 'numpy')
        Exact1 = sympy.lambdify((x, y, z, t), me1, 'numpy')
        Exact2 = sympy.lambdify((x, y, z, t), me2, 'numpy')

        DExact0 = sympy.lambdify((x, y, z, t), mexx0, 'numpy')
        DExact1 = sympy.lambdify((x, y, z, t), mexx1, 'numpy')
        DExact2 = sympy.lambdify((x, y, z, t), mexx2, 'numpy')

        Exact_t0 = sympy.lambdify((x, y, z, t), met0, 'numpy')
        Exact_t1 = sympy.lambdify((x, y, z, t), met1, 'numpy')
        Exact_t2 = sympy.lambdify((x, y, z, t), met2, 'numpy')

        def Exact(p, tn):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            Value = np.zeros((3,) + x.shape, dtype = np.float64)
            Value[0, :] = Exact0(x, y, z, tn)
            Value[1, :] = Exact1(x, y, z, tn)
            Value[2, :] = Exact2(x, y, z, tn)
            return Value

        def Exact_xx(p, tn):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            Value = np.zeros((3,) + x.shape, dtype = np.float64)
            Value[0, :] = DExact0(x, y, z, tn)
            Value[1, :] = DExact1(x, y, z, tn)
            Value[2, :] = DExact2(x, y, z, tn)
            return Value

        def Exact_t(p, tn):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            Value = np.zeros((3,) + x.shape, dtype = np.float64)
            Value[0, :] = Exact_t0(x, y, z, tn)
            Value[1, :] = Exact_t1(x, y, z, tn)
            Value[2, :] = Exact_t2(x, y, z, tn)
            return Value


        self.ExactFun = Exact        
        self.Exact_tFun = Exact_t
        self.Exact_xxFun = Exact_xx
        
        def ForcingForLLG(p, tn):
            me = self.ExactFun(p, tn)
            met = self.Exact_tFun(p, tn)
            mexx = self.Exact_xxFun(p, tn)
            F = met.reshape(-1) + self.OutProduct(me, mexx) - self.alpha * self.OutProduct(me, met)
            if 0 in self.TypeEffective:
                Ani = np.broadcast_to(self.AnisotropyDirection[:, None], shape = me.shape)
                F +=  (self.OutProduct(me, Ani).reshape(3, -1) * np.einsum('qj, qj->j', me, Ani)).reshape(-1)
            return F

        # def ForcingForLL(p, tn):
        #     me = self.ExactFun(p, tn)
        #     met = self.Exact_tFun(p, tn)
        #     mexx = self.Exact_xxFun(p, tn)
        #     meXmexx = self.OutProduct(me, mexx)
        #     F = met.reshape(-1) + meXmexx + self.alpha * self.OutProduct(me, meXmexx)
        #     if 0 in self.TypeEffective:
        #         raise print('This part have not been finished!')
        #     return F

        if self.ATExampleType == "LLG":
            self.ForcingItem = ForcingForLLG
            print("The Example of LLG is set!")
        # elif self.ATExampleType == "LL":
        #     self.ForcingItem = ForcingForLL
        #     print("The Example of LL is set!")

        self.Ms = 1
        self.mu0 = 1
        self.Cex = 1/2
        if 0 in self.TypeEffective:
            self.TypeEffective = (0, 1,)
        else:
            self.TypeEffective = (1,)
        self.Ku = 1/2
        print('self.TypeEffective is set to', self.TypeEffective)
        print('Ms, mu0, Cex, Ku are set to', self.Ms, self.mu0, self.Cex, self.Ku)
        return True


    def AccuracyTestExample2(self):
        import sympy
        x = sympy.Symbol('x')
        y = sympy.Symbol('y')
        z = sympy.Symbol('z')
        t = sympy.Symbol('t')


        Scale_Lin = ((self.SpaceDomain[1] - self.SpaceDomain[0])*(self.SpaceDomain[3] - self.SpaceDomain[2])*(self.SpaceDomain[5] - self.SpaceDomain[4])/8)**4
        # me0 = sympy.cos(x**2 * (self.SpaceDomain[1]-x)**2 * y**2 * (self.SpaceDomain[3]-y)**2 * z**2 * (self.SpaceDomain[5]-z)**2 /Scale_Lin) * sympy.sin(t/self.Deltat/self.gamma/self.Ms/self.MaxNumStep*np.pi/2)
        # me1 = sympy.sin(x**2 * (self.SpaceDomain[1]-x)**2 * y**2 * (self.SpaceDomain[3]-y)**2 * z**2 * (self.SpaceDomain[5]-z)**2 /Scale_Lin) * sympy.sin(t/self.Deltat/self.gamma/self.Ms/self.MaxNumStep*np.pi/2)
        # me2 = sympy.cos(t/self.Deltat/self.gamma/self.Ms/self.MaxNumStep*np.pi/2) + 0*x
        me0 = sympy.cos(x**2 * (self.SpaceDomain[1]-x)**2 * y**2 * (self.SpaceDomain[3]-y)**2 * z**2 * (self.SpaceDomain[5]-z)**2 /Scale_Lin) * sympy.sin(t/self.MaxNumStep*np.pi/2)
        me1 = sympy.sin(x**2 * (self.SpaceDomain[1]-x)**2 * y**2 * (self.SpaceDomain[3]-y)**2 * z**2 * (self.SpaceDomain[5]-z)**2 /Scale_Lin) * sympy.sin(t/self.MaxNumStep*np.pi/2)
        me2 = sympy.cos(t/self.MaxNumStep*np.pi/2) + 0*x

        mexx0 = sympy.diff(me0, x, 2) + sympy.diff(me0, y, 2) + sympy.diff(me0, z, 2)
        mexx1 = sympy.diff(me1, x, 2) + sympy.diff(me1, y, 2) + sympy.diff(me1, z, 2)
        mexx2 = sympy.diff(me2, x, 2) + sympy.diff(me2, y, 2) + sympy.diff(me2, z, 2)

        met0 = sympy.diff(me0, t)
        met1 = sympy.diff(me1, t)
        met2 = sympy.diff(me2, t)

        Exact0 = sympy.lambdify((x, y, z, t), me0, 'numpy')
        Exact1 = sympy.lambdify((x, y, z, t), me1, 'numpy')
        Exact2 = sympy.lambdify((x, y, z, t), me2, 'numpy')

        DExact0 = sympy.lambdify((x, y, z, t), mexx0, 'numpy')
        DExact1 = sympy.lambdify((x, y, z, t), mexx1, 'numpy')
        DExact2 = sympy.lambdify((x, y, z, t), mexx2, 'numpy')

        Exact_t0 = sympy.lambdify((x, y, z, t), met0, 'numpy')
        Exact_t1 = sympy.lambdify((x, y, z, t), met1, 'numpy')
        Exact_t2 = sympy.lambdify((x, y, z, t), met2, 'numpy')

        def Exact(p, tn):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            Value = np.zeros((3,) + x.shape, dtype = np.float64)
            Value[0, :] = Exact0(x, y, z, tn)
            Value[1, :] = Exact1(x, y, z, tn)
            Value[2, :] = Exact2(x, y, z, tn)
            return Value

        def Exact_xx(p, tn):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            Value = np.zeros((3,) + x.shape, dtype = np.float64)
            Value[0, :] = DExact0(x, y, z, tn)
            Value[1, :] = DExact1(x, y, z, tn)
            Value[2, :] = DExact2(x, y, z, tn)
            return Value

        def Exact_t(p, tn):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            Value = np.zeros((3,) + x.shape, dtype = np.float64)
            Value[0, :] = Exact_t0(x, y, z, tn)
            Value[1, :] = Exact_t1(x, y, z, tn)
            Value[2, :] = Exact_t2(x, y, z, tn)
            return Value


        self.ExactFun = Exact        
        self.Exact_tFun = Exact_t
        self.Exact_xxFun = Exact_xx
        
        def ForcingForLLG(p, tn):
            me = self.ExactFun(p, tn)
            met = self.Exact_tFun(p, tn)
            mexx = self.Exact_xxFun(p, tn)
            F = met.reshape(-1) 
            if self.alpha != 0:
                F += - self.alpha * self.OutProduct(me, met)
            if 1 in self.TypeEffective:
                F += self.CofOfExchange *self.OutProduct(me, mexx) 
            if 0 in self.TypeEffective:
                Ani = np.broadcast_to(self.AnisotropyDirection[:, None], shape = me.shape)
                F +=  self.CofOfAnisotropy * (self.OutProduct(me, Ani).reshape(3, -1) * np.einsum('qj, qj->j', me, Ani)).reshape(-1)
            return F




        if self.ATExampleType == "LLG":
            self.ForcingItem = ForcingForLLG
            print("The Example of LLG is set!")
        # elif self.ATExampleType == "LL":
        #     self.ForcingItem = ForcingForLL
        #     print("The Example of LL is set!")

        return True




