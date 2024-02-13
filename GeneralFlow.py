import numpy as np
import pandas as pd
import scipy.integrate as spint
import scipy.optimize as spy
import fluids

# TODO: You can implement TP CEA solver to get thermochemistry data for changing gamma

class GeneralFlow:
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

    def setIC(self, M1, vel, P01, P1, T01, T1, rho, visc, epsilon):
        # TODO: Update so that we have a changing viscosity
        self.visc = visc
        # TODO: Updated for option of changing epsilon
        self.epsilon = epsilon
        self.IC = np.array([M1, vel, P01, P1, T01, T1, rho])

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

    @staticmethod
    def findInfCoeffMach(gamma, M):
        '''
        Returns Influence Coefficients for Mach Number
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

        dM = {'dMdA': dMdA,
              'dMdH0': dMdH0,
              'dMdf': dMdf,
              'dMdmdot': dMdmdot,
              'dMdMW': dMdMW,
              'dMdgamma': dMdgamma}
        return dM

    @staticmethod
    def findInfCoeffVel(gamma, M):
        '''
        Returns Influence Coefficients for Velocity
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

        dV = {'dVdA': dVdA,
              'dVdH0': dVdH0,
              'dVdf': dVdf,
              'dVdmdot': dVdmdot,
              'dVdMW': dVdMW,
              'dVdgamma': dVdgamma}
        return dV
    
    @staticmethod
    def findInfCoeffa(gamma, M):
        '''
        Returns Influence Coefficients for Change in Speed of Sound
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

        da = {'dadA': dadA,
              'dadH0': dadH0,
              'dadf': dadf,
              'dadmdot': dadmdot,
              'dadMW': dadMW,
              'dadgamma': dadgamma}
        return da
    
    @staticmethod
    def findInfCoeffT(gamma, M):
        '''
        Returns Influence Coefficients for Change in Static Temperature
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

        dT = {'dTdA': dTdA,
              'dTdH0': dTdH0,
              'dTdf': dTdf,
              'dTdmdot': dTdmdot,
              'dTdMW': dTdMW,
              'dTdgamma': dTdgamma}
        return dT

    @staticmethod
    def findInfCoeffrho(gamma, M):
        '''
        Returns Influence Coefficients for Change in Density
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

        drho = {'drhodA': drhodA,
                'drhodH0': drhodH0,
                'drhodf': drhodf,
                'drhodmdot': drhodmdot,
                'drhodMW': drhodMW,
                'drhodgamma': drhodgamma}
        return drho
    
    @staticmethod
    def findInfCoeffP(gamma, M):
        '''
        Returns Influence Coefficients for Change in Pressure
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

        dP = {'dPdA': dPdA,
              'dPdH0': dPdH0,
              'dPdf': dPdf,
              'dPdmdot': dPdmdot,
              'dPdMW': dPdMW,
              'dPdgamma': dPdgamma}
        return dP
    
    @staticmethod
    def findInfCoeffF(gamma, M):
        '''
        Returns Influence CoefficientS for Change in ??????????????
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

        dF = {'dFdA': dFdA,
              'dFdH0': dFdH0,
              'dFdf': dFdf,
              'dFdmdot': dFdmdot,
              'dFdMW': dFdMW,
              'dFdgamma': dFdgamma}
        return dF
    
    @staticmethod
    def findInfCoeffS(gamma, M):
        '''
        Returns Influence Coefficietns for Chance in Entropy
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

        dS = {'dSdA': dSdA,
              'dSdH0': dSdH0,
              'dSdf': dSdf,
              'dSdmdot': dSdmdot,
              'dSdMW': dSdMW,
              'dSdgamma': dSdgamma}
        return dS

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

    def xdotSolve(self, t, x):
        # Makes Variables easier to use
        M = x[0]
        V = x[1]
        P0 = x[2]
        P = x[3]
        T01 = x[4]
        T = x[5]
        rho = x[6]

        # Determines if xdot array needs to be overwritten
        if abs(M - self.MCrit) <= self.MCritRange:
            xdot = self.xdotOverride()
            return xdot

        # Gets current Diameter and Current Diameter derivative
        D, dD_dx = self.getD(t)
        
        # Solves for friction factor
        f = self.getf(rho, V, D)

        # Area Calculations
        area = (1 / 4) * np.pi * np.power(D, 2)
        dA_dx = 2 * area / D * dD_dx

        # Stagnation Temperature
        dT0_dx = T01 * self.q / self.Cp

        # Mach Number (this will drive the rest of the derivations)
        # Done
        num1Area = -M * (1 + (self.gamma - 1) / 2 * np.power(M, 2))
        den1Area = area * (1 - np.power(M, 2))
        num1 = 4 * f * self.gamma * np.power(M, 3) * (1 + ((self.gamma - 1) / 2) * np.power(M, 2))
        den1 = 2 * D * (1 - np.power(M, 2))
        dM_dx = dA_dx * num1Area / den1Area + num1 / den1
        # xdot1 = num1 / den1

        # Velocity
        # Done
        num2 = dM_dx / M
        den2 = 1 + (self.gamma - 1) / 2 * np.power(M, 2)
        dV_dx =  V * num2 / den2

        # Stagnation Pressure
        # Done
        part31 = - self.gamma * np.power(M, 2) * 4 * f / (2 * D)
        part32 = - self.gamma * np.power(M, 2) * dT0_dx / (2 * T01)
        dP0_dx = P0 * (part31 + part32)
        # xdot3 = 0

        # Static Pressure
        # Done
        num4 = -self.gamma * M * dM_dx
        den4 = 1 + (self.gamma - 1) / 2 * np.power(M, 2)
        dP_dx = P * (dP0_dx / P0 + num4 / den4)

        # Temperature
        # Done
        num6 = - (self.gamma - 1) * M * dM_dx
        den6 = 1 + (self.gamma - 1) / 2 * np.power(M, 2)
        dT_dx = T * num6 / den6

        # Density
        # Done for now
        C7 = 2 / (1 - np.power(M, 2))
        part71 = (np.power(M, 2) / 2) * (dA_dx / area) - (4 * f * self.gamma * np.power(M, 2)) / (4 * D) - (dT0_dx / (2 * T))
        drho_dx = 0

        self.dM_dx = dM_dx
        self.dV_dx = dV_dx
        self.dP0_dx = dP0_dx
        self.dP_dx = dP_dx
        self.dT0_dx = dT0_dx
        self.dT_dx = dT_dx
        self.drho_dx = drho_dx

        xdot = np.hstack((dM_dx, dV_dx, dP0_dx, dP_dx, dT0_dx, dT_dx, drho_dx))
        return xdot    
    
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
    
    def xdotSolveBeta(self, t, x):
        # TODO: Add Support for Entropy and Impulse
        # Makes Variables easier to use
        M2 = x[0]
        # print(M2)
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

        # TODO: Add support for finding current gamma
        gamma = self.gamma

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
        dM2_dx = dMInf['dMdA'] * dA_dx / area + \
                 dMInf['dMdH0'] * dH0_dx / Cp / T + \
                 dMInf['dMdf'] * f * 4. / D + \
                 dMInf['dMdf'] * 2 * dX / gamma / P / area / M**2 + \
                 dMInf['dMdf'] * -2 * vgx / vg / mdot * dmdot_dx + \
                 dMInf['dMdmdot'] * dmdot_dx + \
                 dMInf['dMdMW'] * dMW_dx + \
                 dMInf['dMdgamma'] * dgamma_dx
        dM2_dx *= M2
        
        # Calculates Derivative for Velocity
        dV_dx = dVInf['dVdA'] * dA_dx / area + \
                dVInf['dVdH0'] * dH0_dx / Cp / T + \
                dVInf['dVdf'] * f * 4. / D + \
                dVInf['dVdf'] * 2 * dX / gamma / P / area / M**2 + \
                dVInf['dVdf'] * -2 * vgx / vg / mdot * dmdot_dx + \
                dVInf['dVdmdot'] * dmdot_dx + \
                dVInf['dVdMW'] * dMW_dx + \
                dVInf['dVdgamma'] * dgamma_dx
        dV_dx *= V
        
        # Calculates Derivative for Speed of Sound
        da_dx = daInf['dadA'] * dA_dx / area + \
                daInf['dadH0'] * dH0_dx / Cp / T + \
                daInf['dadf'] * f * 4. / D + \
                daInf['dadf'] * 2 * dX / gamma / P / area / M**2 + \
                daInf['dadf'] * -2 * vgx / vg / mdot * dmdot_dx + \
                daInf['dadmdot'] * dmdot_dx + \
                daInf['dadMW'] * dMW_dx + \
                daInf['dadgamma'] * dgamma_dx
        da_dx *= a
        
        # Calculates Derivative for Temperature
        dT_dx = dTInf['dTdA'] * dA_dx / area + \
                dTInf['dTdH0'] * dH0_dx / Cp / T + \
                dTInf['dTdf'] * f * 4. / D + \
                dTInf['dTdf'] * 2 * dX / gamma / P / area / M**2 + \
                dTInf['dTdf'] * -2 * vgx / vg / mdot * dmdot_dx + \
                dTInf['dTdmdot'] * dmdot_dx + \
                dTInf['dTdMW'] * dMW_dx + \
                dTInf['dTdgamma'] * dgamma_dx
        dT_dx *= T

        # Calculates Derivative for Density
        drho_dx = drhoInf['drhodA'] * dA_dx / area + \
                  drhoInf['drhodH0'] * dH0_dx / Cp / T + \
                  drhoInf['drhodf'] * f * 4. / D + \
                  drhoInf['drhodf'] * 2 * dX / gamma / P / area / M**2 + \
                  drhoInf['drhodf'] * -2 * vgx / vg / mdot * dmdot_dx + \
                  drhoInf['drhodmdot'] * dmdot_dx + \
                  drhoInf['drhodMW'] * dMW_dx + \
                  drhoInf['drhodgamma'] * dgamma_dx
        drho_dx *= rho

        # Calculates Derivative for Pressure
        dP_dx = dPInf['dPdA'] * dA_dx / area + \
                dPInf['dPdH0'] * dH0_dx / Cp / T + \
                dPInf['dPdf'] * f * 4. / D + \
                dPInf['dPdf'] * 2 * dX / gamma / P / area / M**2 + \
                dPInf['dPdf'] * -2 * vgx / vg / mdot * dmdot_dx + \
                dPInf['dPdmdot'] * dmdot_dx + \
                dPInf['dPdMW'] * dMW_dx + \
                dPInf['dPdgamma'] * dgamma_dx
        dP_dx *= P


        # if self.chokepoint == False and (M - (self.MCrit + 0.01)):
        self.dM2_dx = dM2_dx
        self.dV_dx = dV_dx
        self.da_dx = da_dx
        self.dT_dx = dT_dx
        self.drho_dx = drho_dx
        self.dP_dx = dP_dx
        # self.chokepoint = True
        print(M)

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
                  max_step=0.001)
        
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


    def run(self):
        '''
        Runs Solver on ititial conditions for engine
        '''
        self.results = spint.solve_ivp(fun=self.xdotSolve, 
                  t_span=[self.xStart, self.xEnd], 
                  y0=self.IC,
                  method='RK45', 
                  t_eval=self.xspan, 
                  rtol=1e-10,
                  atol=1e-10)
        
        if len(self.results.y) == 0:
            raise ValueError('Convergence Failed')
        
        xarr = self.results.t
        Mach = self.results.y[0]
        Vel = self.results.y[1]
        P0 = self.results.y[2]
        P = self.results.y[3]
        T0 = self.results.y[4]
        T = self.results.y[5]
        rho = self.results.y[6]
        dictOut = {'x': xarr,
                   'Mach': Mach,
                   'Vel': Vel,
                   'P0': P0,
                   'P': P,
                   'T0': T0,
                   'T': T,
                   'rho': rho}
        
        self.results = pd.DataFrame(dictOut)
        return self.results

if __name__ == '__main__':
    results = GeneralFlow.findInfCoeffS(1.4, 2.)
    for key, item in results.items():
        print(f'{key}: {item:.3f}')
