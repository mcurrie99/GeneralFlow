import numpy as np
import pandas as pd
import scipy.integrate as spint
import scipy.optimize as spy
import numba as nb
import CEA_Wrap as CEA
import fluids

# TODO: You can implement TP CEA solver to get thermochemistry data for changing gamma

class GeneralFlow:
    '''
    This is a general Class for a working General Flow Simulation of Flow through a nozzle.
    As this is early in development, working with the class to obtain the results will most likely be necessary.
    '''
    def __init__(self, x_arr=[], D_arr=[], pointNum=1000, q=0., Cp=1005, gamma=1.4, MW=28.96):
        self.xStart = x_arr[0]
        self.xEnd = x_arr[-1]
        self.xLen = pointNum
        self.xspan = np.linspace(self.xStart, self.xEnd, self.xLen)
        self.x_arr = np.array(x_arr)
        self.D_arr = np.array(D_arr)
        self.q = q
        self.Cp = Cp
        self.gamma = gamma
        self.MW = MW
        self.MCrit = 1.0
        self.MCritRange = 0.001
        self.solved = False
        self.chokepoint = False
        self.tCEAPrev = None
        self.maxStep = 0.001
        self.runCEA = False

    def getD(self, t):
        index = np.searchsorted(self.x_arr, t)
        if index < len(self.x_arr):
            if self.x_arr[index] == t:
                D = self.D_arr[index]
                dD_dx = (self.D_arr[index] - self.D_arr[index-1]) / (self.x_arr[index] - self.x_arr[index-1]) if index != 0 else 0
            else:
                D = self.D_arr[index-1] + (t - self.x_arr[index-1]) * (self.D_arr[index] - self.D_arr[index-1]) / (self.x_arr[index] - self.x_arr[index-1])
                dD_dx = (self.D_arr[index] - self.D_arr[index-1]) / (self.x_arr[index] - self.x_arr[index-1]) if index != 0 else 0
        else:
            D = self.D_arr[-1]
            index = len(self.x_arr) - 1
            dD_dx = 0

        return D, dD_dx
    
    def xdotOverride(self):
        xdot = np.hstack((self.dM2_dx, self.dV_dx, self.da_dx, self.dT_dx, self.drho_dx, self.dP_dx))
        return xdot
        
    
    def getf(self, rho, V, D):
        # TODO: Add ability to grab viscosity changes
        Re = rho * V * D / self.visc
        # frictionFactorFunc = lambda f : -4.0 * np.log10(self.epsilon / (3.7 * D) + 1.255 / (Re * np.sqrt(f))) - 1 / np.sqrt(f)
        # fricFactorPenalty = lambda f, init_guess : frictionFactorFunc(f) + np.power(max([0, init_guess - f]), 2)
        # f = spy.fsolve(func=frictionFactorFunc, 
        #                x0=0.0001)
        # This does not use the classic solver approach but is close enough
        # as numerically solving natural logs suck
        f = fluids.friction_factor(Re = Re, eD=self.epsilon/D) / 4.
        return f

    # Sets up CEA Calculations
    def setCEAProp(self, ox:str, oxTemp:float, fu:str, fuTemp:float, OF:float):
        '''
        Sets up Oxidizer and Fuel Materials for CEA Simulations
        Temperature is in units of Kelvin
        '''
        # Oxidizer Material for CEA Setup
        self.Ox = CEA.Oxidizer(name=ox,
                               temp=oxTemp)
        
        # Fuel Material for CEA Setup
        self.Fu = CEA.Fuel(name=fu,
                           temp=fuTemp)

        # Class Setup of OF Ratio
        self.OF = OF
        self.runCEA = True

    # Perfroms Chemical Analysis
    def thermochem(self, temp, press):
        '''
        Returns the derivative of Molecular Weight and Ratio of Specific Heat
        '''
        pressBar = press / 100000.

        problem = CEA.TPProblem(materials=[self.Ox, self.Fu],
                                o_f=self.OF,
                                temperature=temp,
                                temperature_units='K',
                                pressure=pressBar,
                                pressure_units='bar')
        
        results = problem.run()

        gamma = results.gamma
        MW = results.mw
        return gamma, MW

    @staticmethod
    @nb.njit
    def findInfCoeffMach(gamma, M):
        '''
        Returns Influence Coefficients for Mach Number

        Return Array Index Meanings:
        - dMdA: 0
        - dMdH0: 1
        - dMdf: 2
        - dMdmdot: 3
        - dMdM2: 4
        - dMdgamma: 5
        '''
        # Influence Coefficient for Area Change
        dMdA = (-2 * (1 + ((gamma - 1) / 2) * M**2)) / \
               (1 - M**2)
        
        # Influence Coefficient for Stag Enthalpy Change
        dMdH0 = (1 + gamma * M**2) / (1 - M**2)

        # Influence Coefficient for Forces
        dMdfNum = gamma * M**2 * (1 + ((gamma - 1) / 2) * M**2)
        dMdfDen = 1 - M**2
        dMdf = dMdfNum / dMdfDen

        # Influence Coefficient for Added Mass Flow
        dMdmdotNum = 2 * (1 + gamma * M**2) * (1 + ((gamma - 1) / 2) * M**2)
        dMdmdotDen = 1 - M**2
        dMdmdot = dMdmdotNum / dMdmdotDen

        # Influence Coefficient for Work
        dMdMWNum = - (1 + gamma * M**2)
        dMdMWDen = (1 - M**2)
        dMdMW = dMdMWNum / dMdMWDen

        # Influence Coefficient for Gamma Change
        dMdgamma = -1.

        dM = np.array([dMdA,
                       dMdH0,
                       dMdf,
                       dMdmdot,
                       dMdMW,
                       dMdgamma])
        return dM

    @staticmethod
    @nb.njit
    def findInfCoeffVel(gamma, M):
        '''
        Returns Influence Coefficients for Velocity

        Return Array Index Meanings:
        - dVdA: 0
        - dVdH0: 1
        - dVdf: 2
        - dVdmdot: 3
        - dVdM2: 4
        - dVdgamma: 5
        '''
        # Influence Coefficient for Area Change
        dVdA = -1 / (1 - M**2)

        # Influence Coefficient for Stag Enthalpy Change
        dVdH0 = 1 / (1 - M**2)

        # Influence Coefficient for Forces
        dVdf = (gamma * M**2) / (2 * (1 - M**2))

        # Influence Coefficient for Added Mass Flow
        dVdmdot = (1 + gamma * M**2) / (1 - M**2)

        # Influence Coefficient for Work
        dVdMW = -1 / (1 - M**2)

        # Influence Coefficient for Gamma Change
        dVdgamma = 0

        dV = np.array([dVdA,
                       dVdH0,
                       dVdf,
                       dVdmdot,
                       dVdMW,
                       dVdgamma])
        return dV
    
    @staticmethod
    @nb.njit
    def findInfCoeffa(gamma, M):
        '''
        Returns Influence Coefficients for Change in Speed of Sound

        Return Array Index Meanings:
        - dadA: 0
        - dadH0: 1
        - dadf: 2
        - dadmdot: 3
        - dadM2: 4
        - dadgamma: 5
        '''
        # Influence Coefficient for Area Change
        dadANum = ((gamma - 1) / 2) * M**2
        dadADen = (1 - M**2)
        dadA = dadANum / dadADen

        # Influence Coefficient for Stag Enthalpy Change
        dadH0Num = 1 - gamma * M**2
        dadH0Den = 2 * (1 - M**2)
        dadH0 = dadH0Num / dadH0Den

        # Influence Coefficient for Forces
        dadfNum = -gamma * (gamma - 1) * M**4
        dadfDen = 4 * (1 - M**2)
        dadf = dadfNum / dadfDen

        # Influence Coefficient for Added Mass Flow
        dadmdotNum = -((gamma - 1) / 2) * M**2 * (1 + gamma * M**2)
        dadmdotDen = 1 - M**2
        dadmdot = dadmdotNum / dadmdotDen

        # Influence Coefficient for Work
        dadMWNum = gamma * M**2 - 1
        dadMWDen = 2 * (1 - M**2)
        dadMW = dadMWNum / dadMWDen

        # Influence Coefficient for Gamma Change
        dadgamma = 1 / 2

        da = np.array([dadA,
                       dadH0,
                       dadf,
                       dadmdot,
                       dadMW,
                       dadgamma])
        return da
    
    @staticmethod
    @nb.njit
    def findInfCoeffT(gamma, M):
        '''
        Returns Influence Coefficients for Change in Static Temperature

        Return Array Index Meanings:
        - dTdA: 0
        - dTdH0: 1
        - dTdf: 2
        - dTdmdot: 3
        - dTdM2: 4
        - dTdgamma: 5
        '''
        # Influence Coefficient for Area Change
        dTdANum = (gamma - 1) * M**2
        dTdADen = 1 - M**2
        dTdA = dTdANum / dTdADen

        # Influence Coefficient for Stag Enthalpy Change
        dTdH0Num = 1 - gamma * M**2
        dTdH0Den = 1 - M**2
        dTdH0 = dTdH0Num / dTdH0Den

        # Influence Coefficient for Forces
        dTdfNum = -gamma * (gamma - 1) * M**4
        dTdfDen = 2 * (1 - M**2)
        dTdf = dTdfNum / dTdfDen

        # Influence Coefficient for Added Mass Flow
        dTdmdotNum = -(gamma - 1) * M**2 * (1 + gamma * M**2)
        dTdmdotDen = 1 - M**2
        dTdmdot = dTdmdotNum / dTdmdotDen

        # Influence Coefficient for Work
        dTdMWNum = (gamma - 1) * M**2
        dTdMWDen = 1 - M**2
        dTdMW = dTdMWNum / dTdMWDen

        # Influence Coefficient for Gamma Change
        dTdgamma = 0

        dT = np.array([dTdA,
                       dTdH0,
                       dTdf,
                       dTdmdot,
                       dTdMW,
                       dTdgamma])
        return dT

    @staticmethod
    @nb.njit
    def findInfCoeffrho(gamma, M):
        '''
        Returns Influence Coefficients for Change in Density

        Return Array Index Meanings:
        - drhodA: 0
        - drhodH0: 1
        - drhodf: 2
        - drhodmdot: 3
        - drhodM2: 4
        - drhodgamma: 5
        '''
        # Influence Coefficient for Area Change
        drhodA = M**2 / (1 - M**2)

        # Influence Coefficient for Stag Enthalpy Change
        drhodH0 = -1 / (1 - M**2)

        # Influence Coefficient for Forces
        drhodf = -gamma * M**2 / (2 * (1 - M**2))

        # Influence Coefficient for Added Mass Flow
        drhodmdot = -(gamma + 1) * M**2 / (1 - M**2)

        # Influcence Coefficient for Work
        drhodMW = 1 / (1 - M**2)

        # Influence Coefficient for Gamma Change
        drhodgamma = 0

        drho = np.array([drhodA,
                         drhodH0,
                         drhodf,
                         drhodmdot,
                         drhodMW,
                         drhodgamma])
        return drho
    
    @staticmethod
    @nb.njit
    def findInfCoeffP(gamma, M):
        '''
        Returns Influence Coefficients for Change in Pressure

        Return Array Index Meanings:
        - dPdA: 0
        - dPdH0: 1
        - dPdf: 2
        - dPdmdot: 3
        - dPdM2: 4
        - dPdgamma: 5
        '''
        # Influence Coefficient for Area Change
        dPdA = gamma * M**2 / (1 - M**2)

        # Influence Coefficient for Stag Enthalpy Change
        dPdH0 = -gamma * M**2 / (1 - M**2)

        # Influence Coefficient for Forces
        dPdfNum = -gamma * M**2 * (1 + (gamma - 1) * M**2)
        dPdfDen = 2 * (1 - M**2)
        dPdf = dPdfNum / dPdfDen

        # Influence Coefficient for Added Mass Flow
        dPdmdotNum = -2 * gamma * M**2 * (1 + (gamma - 1) / 2 * M**2)
        dPdmdotDen = 1 - M**2
        dPdmdot = dPdmdotNum / dPdmdotDen

        # Influence Coefficient for Work
        dPdMW = gamma * M**2 / (1 - M**2)

        # Influence Coefficient for Gamma Change
        dPdgamma = 0.

        dP = np.array([dPdA,
                       dPdH0,
                       dPdf,
                       dPdmdot,
                       dPdMW,
                       dPdgamma])
        return dP
    
    @staticmethod
    @nb.njit
    def findInfCoeffF(gamma, M):
        '''
        Returns Influence CoefficientS for Change in Impulse Function

        Return Array Index Meanings:
        - dFdA: 0
        - dFdH0: 1
        - dFdf: 2
        - dFdmdot: 3
        - dFdM2: 4
        - dFdgamma: 5
        '''
        # Influence Coefficient for Change in Area
        dFdA = 1 / (1 + gamma * M**2)

        # Influecne Coefficient for Change in Stag Enthalpy Change
        dFdH0 = 0

        # Influence Coefficient for Change in Forces
        dFdf = -gamma * M**2 / (2 * (1 + gamma * M**2))

        # Influence Coefficient for Added Mass Flow
        dFdmdot = 0.

        # Influence Coefficient for Work
        dFdMW = 0.

        # Influence Coefficient for Gamma Change
        dFdgamma = 0.

        dF = np.array([dFdA,
                       dFdH0,
                       dFdf,
                       dFdmdot,
                       dFdMW,
                       dFdgamma])
        return dF
    
    @staticmethod
    @nb.njit
    def findInfCoeffS(gamma, M):
        '''
        Returns Influence Coefficietns for Chance in Entropy

        Return Array Index Meanings:
        - dSdA: 0
        - dSdH0: 1
        - dSdf: 2
        - dSdmdot: 3
        - dSdM2: 4
        - dSdgamma: 5
        '''
        # Influence Coefficient for Change in Area
        dSdA = 0.

        # Influence Coefficient for Stag Enthalpy Change
        dSdH0 = 1.

        # Influence Coefficient for Forces
        dSdf = (gamma - 1) * M**2 / 2

        # Influence Coefficient for Added Mass Flow
        dSdmdot = (gamma - 1) * M**2

        # Influence Coefficeint for Work
        dSdMW = 0.

        # Influence Coefficient for Gamma Change
        dSdgamma = 0.

        dS = np.array([dSdA,
                       dSdH0,
                       dSdf,
                       dSdmdot,
                       dSdMW,
                       dSdgamma])
        return dS

    # TODO: Convert this to numba
    @staticmethod
    def findInfCoeffP0(gamma, M):
        '''
        Returns influence Coefficients for Stagnation Pressure
        ***
        This is for Constant Ratio of Specific Heats and Constant Molecular Weight Only
        ***
        '''
        # Influence Coefficient for Change in Area
        dP0dA = 0

        # Influence Coefficient for Change in Stag Enthalpy
        dP0dH0 = -gamma * M**2 / 2

        # Influence Coefficient for Forces
        dP0df = -gamma * M**2 / 2

        # Influence Coefficient for Added Mass Flow
        dP0dmdot = -gamma * M**2

        dP0 = {'dP0dA': dP0dA,
               'dP0dH0': dP0dH0,
               'dP0df': dP0df,
               'dP0dmdot': dP0dmdot}
        return dP0
    
    def findThermoDervs(self, T, P, t):
        '''
        Returns dgamma/dx and dMW/dx
        '''
        # CEA Run for further Calculations and Prepares for delta Calculations
        gammaCurr, MWCurr = self.thermochem(T, P)

        # Grabs data and ensures value is largest seen so far
        if self.tCEAPrev == None:
            # Sets initial Derivative
            dMW_dx = 0.
            dgamma_dx = 0.

            self.gammaCEAPrev = gammaCurr
            self.MWCEAPrev = MWCurr
            self.tCEAPrev = t
        else:          
            tDiff = t - self.tCEAPrev

            # Prevents divide by 0 error
            if tDiff == 0.0:
                dgamma_dx = 0.
                dMW_dx = 0.0
            else:
                dgamma_dx = (gammaCurr - self.gammaCEAPrev) / tDiff
                dMW_dx = (MWCurr - self.MWCEAPrev) / tDiff

                self.gammaCEAPrev = gammaCurr
                self.MWCEAPrev = MWCurr
                self.tCEAPrev = t

        return dgamma_dx, dMW_dx, gammaCurr
    
    def setICBeta(self, M1, vel, P1, T1, rho1, visc, epsilon):
        # TODO: Update so that we have a changing viscosity
        self.visc = visc

        # TODO: Updated for option of changing epsilon
        self.epsilon = epsilon

        # TODO: Update so that we can have a changing Gamma and MW
        gamma = self.gamma
        MW = self.MW
        a = np.sqrt(gamma * 8314 * T1 / self.MW)

        # Sets Initial Conditions
        self.IC = np.array([M1**2, vel, a, T1, rho1, P1])
    
    # @nb.jit(nopython=True)
    def xdotSolveBeta(self, t, x):
        # TODO: Add Support for Entropy and Impulse
        # Makes Variables easier to use
        M2 = x[0]
        M = np.sqrt(M2)
        V = x[1]
        a = x[2]
        T = x[3]
        rho = x[4]
        P = x[5]

        # TODO: Remove this correction
        if t > 0.0 and M < 1.:
            M = 1.005
            M2 = M**2

            
        # Grabs Derivative for gamma and MW and current Values
        if self.runCEA == True:
            dgamma_dx, dMW_dx, gamma = self.findThermoDervs(T, P, t)

        # Grabs Influence Coefficients
        dMInf = self.findInfCoeffMach(gamma, M)
        dVInf = self.findInfCoeffVel(gamma, M)
        daInf = self.findInfCoeffa(gamma, M)
        dTInf = self.findInfCoeffT(gamma, M)
        drhoInf = self.findInfCoeffrho(gamma, M)
        dPInf = self.findInfCoeffP(gamma, M)
        # These are useless right now
        dFInf = self.findInfCoeffF(gamma, M)
        dSInf = self.findInfCoeffS(gamma, M)

        # TODO: Implement Models for each of the of the terms
        # Gets current Diameter and Current Diameter derivative
        D, dD_dx = self.getD(t)
        area = (1 / 4) * np.pi * np.power(D, 2)
        dA_dx = np.pi * D * dD_dx / 2

        # Gets Stagnation Enthalpy Term
        dH0_dx = 0
        Cp = 1

        # Gets External Forces Term
        f = self.getf(rho, V, D)
        # f = 0.
        dX = 0.
        vgx = 0
        vg = 1.

        # Gets Added Mass Flow Rate Term
        mdot = 1.
        dmdot_dx = 0.

        # Gets Change in Molecular Weight
        dMW_dx = 0.

        # Gets Change in Gamma Term
        dgamma_dx = 0.

        # Determines if xdot array needs to be overwritten
        if abs(M - self.MCrit) <= self.MCritRange:
            xdot = self.xdotOverride()
            return xdot

        # Calculates Derivative for Mach Number Squared
        dM2_dx = dMInf[0] * dA_dx / area + \
                 dMInf[1] * dH0_dx / Cp / T + \
                 dMInf[2] * f * 4. / D + \
                 dMInf[2] * 2 * dX / gamma / P / area / M**2 + \
                 dMInf[2] * -2 * vgx / vg / mdot * dmdot_dx + \
                 dMInf[3] * dmdot_dx + \
                 dMInf[4] * dMW_dx + \
                 dMInf[5] * dgamma_dx
        dM2_dx *= M2
        
        # Calculates Derivative for Velocity
        dV_dx = dVInf[0] * dA_dx / area + \
                dVInf[1] * dH0_dx / Cp / T + \
                dVInf[2] * f * 4. / D + \
                dVInf[2] * 2 * dX / gamma / P / area / M**2 + \
                dVInf[2] * -2 * vgx / vg / mdot * dmdot_dx + \
                dVInf[3] * dmdot_dx + \
                dVInf[4] * dMW_dx + \
                dVInf[5] * dgamma_dx
        dV_dx *= V
        
        # Calculates Derivative for Speed of Sound
        da_dx = daInf[0] * dA_dx / area + \
                daInf[1] * dH0_dx / Cp / T + \
                daInf[2] * f * 4. / D + \
                daInf[2] * 2 * dX / gamma / P / area / M**2 + \
                daInf[2] * -2 * vgx / vg / mdot * dmdot_dx + \
                daInf[3] * dmdot_dx + \
                daInf[4] * dMW_dx + \
                daInf[5] * dgamma_dx
        da_dx *= a
        
        # Calculates Derivative for Temperature
        dT_dx = dTInf[0] * dA_dx / area + \
                dTInf[1] * dH0_dx / Cp / T + \
                dTInf[2] * f * 4. / D + \
                dTInf[2] * 2 * dX / gamma / P / area / M**2 + \
                dTInf[2] * -2 * vgx / vg / mdot * dmdot_dx + \
                dTInf[3] * dmdot_dx + \
                dTInf[4] * dMW_dx + \
                dTInf[5] * dgamma_dx
        dT_dx *= T

        # Calculates Derivative for Density
        drho_dx = drhoInf[0] * dA_dx / area + \
                  drhoInf[1] * dH0_dx / Cp / T + \
                  drhoInf[2] * f * 4. / D + \
                  drhoInf[2] * 2 * dX / gamma / P / area / M**2 + \
                  drhoInf[2] * -2 * vgx / vg / mdot * dmdot_dx + \
                  drhoInf[3] * dmdot_dx + \
                  drhoInf[4] * dMW_dx + \
                  drhoInf[5] * dgamma_dx
        drho_dx *= rho

        # Calculates Derivative for Pressure
        dP_dx = dPInf[0] * dA_dx / area + \
                dPInf[1] * dH0_dx / Cp / T + \
                dPInf[2] * f * 4. / D + \
                dPInf[2] * 2 * dX / gamma / P / area / M**2 + \
                dPInf[2] * -2 * vgx / vg / mdot * dmdot_dx + \
                dPInf[3] * dmdot_dx + \
                dPInf[4] * dMW_dx + \
                dPInf[5] * dgamma_dx
        dP_dx *= P


        # if self.chokepoint == False and (M - (self.MCrit + 0.01)):
        self.dM2_dx = dM2_dx
        self.dV_dx = dV_dx
        self.da_dx = da_dx
        self.dT_dx = dT_dx
        self.drho_dx = drho_dx
        self.dP_dx = dP_dx
        # self.chokepoint = True
        # print(M)

        # Brings Derivatives Together and Returns Algorithm
        xdot = np.hstack((dM2_dx, dV_dx, da_dx, dT_dx, drho_dx, dP_dx))
        return xdot
    
    def runBeta(self):
        '''
        Runs Solver on ititial conditions for engine
        '''
        self.results = spint.solve_ivp(fun=self.xdotSolveBeta, 
                  t_span=[self.xStart, self.xEnd], 
                  y0=self.IC,
                  method='RK45', 
                  t_eval=self.xspan, 
                  rtol=1e-10,
                  atol=1e-10,
                  max_step=self.maxStep)
        
        if len(self.results.y) == 0:
            raise ValueError('Convergence Failed')
        
        xarr = self.results.t
        Mach = np.sqrt(np.array(self.results.y[0]))
        Vel = self.results.y[1]
        a = self.results.y[2]
        T = self.results.y[3]
        rho = self.results.y[4]
        P = self.results.y[5]
        self.reason = self.results.message

        dictOut = {'x': xarr,
                   'Mach': Mach,
                   'Vel': Vel,
                   'a': a,
                   'T': T,
                   'rho': rho,
                   'P': P,
                   'reason': self.reason}
        
        self.chokepoint = False
        
        self.results = pd.DataFrame(dictOut)
        return self.results

if __name__ == '__main__':
    results = GeneralFlow.findInfCoeffS(1.4, 2.)
    for i in results:
        print(i)
