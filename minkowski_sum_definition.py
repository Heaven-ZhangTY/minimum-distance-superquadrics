"""
    Define closed-form Minkowski sum of superquadrics
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from function import *

class super_qudric():

    def __init__(self, n1, n2, a, b, c):
        self.n1 = n1
        self.n2 = n2
        self.a = a
        self.b = b
        self.c = c
        self.n = 20

    def def_super_qudric(self, u, v, n1, n2, a, b, c):
        "Define superquadric function"
        # Center point coordinates of the superquadric
        x0 = 0
        y0 = 0
        z0 = 0
        x = x0 + a * np.sign(np.cos(u)) * np.power(np.cos(u), n1) * \
            np.sign(np.cos(v)) * np.power(np.abs(np.cos(v)), n2)
        y = y0 + b * np.sign(np.cos(u)) * np.power(np.abs(np.cos(u)), n1) * \
            np.sign(np.sin(v)) * np.power(np.abs(np.sin(v)), n2)
        z = z0 + c * np.sign(np.sin(u)) * np.power(np.abs(np.sin(u)), n1)
        return x, y, z
    
    def point_super_qudric(self):
        "Generate coordinate points"
        n = self.n 
        u = np.linspace(-np.pi/2, np.pi/2, n)                                                                                    
        v = np.linspace(-np.pi, np.pi, n)
        x, y, z = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x[i, j], y[i, j], z[i, j] = self.def_super_qudric(
                    u[i], v[j], self.n1, self.n2, self.a, self.b, self.c)
        return x, y, z 
    
    def RotationXYZ(self,x,y,z,RotX,RotY,RotZ,TranX,TranY,TranZ):
        n = self.n 
        M = ConvertRPYToMat(RotX,RotY,RotZ)
        # print(M)
        T = np.array([[TranX],[TranY],[TranZ]])
        # print(T)
        x_new, y_new, z_new = np.zeros((n, n)),np.zeros((n, n)),np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x1 = x[i,j]
                y1 = y[i,j]
                z1 = z[i,j]
                x_arr = np.array([[x1],[y1],[z1]])
                x_trans = np.dot(M,x_arr) +  T
                x_new[i,j] = x_trans[0]
                y_new[i,j] = x_trans[1]
                z_new[i,j] = x_trans[2]
        return x_new, y_new, z_new
                        
    # Define: Gradient representation expression
    def def_gradient(self,u, v, n1, n2, a, b, c):
        grad_x = 2 * np.sign(np.cos(u)) * np.power(np.abs(np.cos(u)), 2-n1) * \
            np.sign(np.cos(v)) * np.power(np.abs(np.cos(v)), 2-n2)/(a*n1)
        grad_y = 2 * np.sign(np.cos(u)) * np.power(np.abs(np.cos(u)), 2-n1) * \
            np.sign(np.sin(v)) * np.power(np.abs(np.sin(v)), 2-n2)/(b*n2)
        grad_z = 2 * np.sign(np.sin(u)) * \
            np.power(np.abs(np.sin(u)), 2-n1)/(c*n1)
        return grad_x, grad_y, grad_z

    # Point gradient
    def point_gradient(self): 
        n = self.n 
        u = np.linspace(-np.pi/2, np.pi/2, n)
        v = np.linspace(-np.pi, np.pi, n)
        grad_x, grad_y, grad_z = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_x[i, j], grad_y[i, j], grad_z[i, j] = self.def_gradient(
                    u[i], v[j], self.n1, self.n2, self.a, self.b, self.c)
        return grad_x, grad_y, grad_z

    # Parameterize the superquadric with gradient parameters
    def point_gradient_parameter(self, grad_x, grad_y, grad_z):
        # Superquadric parameters
        n1 = self.n1
        n2 = self.n2
        a = self.a
        b = self.b
        c = self.c
        n = self.n 
        # print(n)
        x, y, z, gama= np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)),np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gama[i, j] = 1 - np.sign(0.5*c*n1*grad_z[i, j]) * np.power(np.abs(0.5*c*n1*grad_z[i, j]), 2/(2-n1))
                x[i, j] = a * np.sign(0.5*a*n1*grad_x[i, j]) * np.power(np.abs(0.5*a*n1*grad_x[i, j]), n2/(2-n1)) * np.sign(gama[i, j]) * np.power(np.abs(gama[i, j]), (n1-n2)/(2-n2))
                y[i, j] = b * np.sign(0.5*b*n1*grad_y[i, j]) * np.power(np.abs(0.5*b*n1*grad_y[i, j]), n2/(2-n1)) * np.sign(gama[i, j]) * np.power(np.abs(gama[i, j]), (n1-n2)/(2-n2))
                z[i, j] = c * np.sign(0.5*c*n1*grad_z[i, j]) * np.power(np.abs(0.5*c*n1*grad_z[i, j]), n1/(2-n1))
        return x, y, z

    # mu
    def mu(self, grad_x_01, grad_y_01, grad_z_01, n1_02, n2_02, a_02, b_02, c_02):
        n1 = n1_02
        n2 = n2_02
        a = a_02
        b = b_02
        c = c_02
        n = self.n
        grad_inv_x, grad_inv_y, grad_inv_z= np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        mu_x, mu_y, mu_z= np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        ro, m, fan = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ro[i,j] = np.sign(0.5*a_02*n1_02*grad_x_01[i,j]) * np.power(np.abs(0.5*a_02*n1_02*grad_x_01[i,j]),2/(2-n2_02)) + \
                          np.sign(0.5*b_02*n1_02*grad_y_01[i,j]) * np.power(np.abs(0.5*b_02*n1_02*grad_y_01[i,j]),2/(2-n2_02))
                m[i,j] = np.sign(ro[i,j])*np.power(np.abs(ro[i,j]),(n1_02-n2_02)/(4-2*n1_02))
                grad_inv_x[i,j] = np.sign(0.5*a_02*n1_02*grad_x_01[i,j])*np.power(np.abs(0.5*a_02*n1_02*grad_x_01[i,j]),1/(2-n2_02)) * m[i,j]
                grad_inv_y[i,j] = np.sign(0.5*b_02*n1_02*grad_y_01[i,j])*np.power(np.abs(0.5*b_02*n1_02*grad_y_01[i,j]),1/(2-n2_02)) * m[i,j]
                grad_inv_z[i,j] = np.sign(0.5*c_02*n1_02*grad_z_01[i,j])*np.power(np.abs(0.5*c_02*n1_02*grad_z_01[i,j]),1/(2-n1_02))
                fan[i,j] =  np.power(grad_inv_x[i,j]**2+grad_inv_y[i,j]**2+grad_inv_z[i,j]**2,0.5)
                mu_x[i,j] = grad_inv_x[i,j] / fan[i,j]
                mu_y[i,j] = grad_inv_y[i,j] / fan[i,j]
                mu_z[i,j] = grad_inv_z[i,j] / fan[i,j]
        return mu_x, mu_y, mu_z
    
    def g(self, mu_x, mu_y, mu_z, n1_02, n2_02, a_02, b_02, c_02):
        n = self.n
        g_x, g_y, g_z= np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(n): 
                m[i,j] = np.sign(mu_x[i,j]**2+mu_y[i,j]**2)*np.power(np.abs(mu_x[i,j]**2+mu_y[i,j]**2),(n2_02-n1_02)/2)
                g_x[i, j] = 2/(a_02*n1_02) * np.sign(mu_x[i,j]) * np.power(np.abs(mu_x[i,j]),(2-n2_02)) * m[i,j]
                g_y[i, j] = 2/(b_02*n1_02) * np.sign(mu_y[i,j]) * np.power(np.abs(mu_y[i,j]),(2-n2_02)) * m[i,j]
                g_z[i, j] = 2/(b_02*n1_02) * np.sign(mu_z[i,j]) * np.power(np.abs(mu_z[i,j]),(2-n1_02))
        return g_x, g_y, g_z

    # Minkowski Sum 
    def MinSum(self, g_x, g_y, g_z, grad_x, grad_y, grad_z):
        n = self.n
        x1, y1, z1 = self.point_gradient_parameter(grad_x, grad_y, grad_z)
        fan, mo, m = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                fan[i,j] = np.power(g_x[i,j]**2 + g_y[i,j]**2 + g_z[i,j]**2, 0.5)
                mo[i,j] = np.power(grad_x[i,j]**2 + grad_y[i,j]**2 + grad_z[i,j]**2, 0.5)
                # fan[i,j] = np.linalg.norm([g_x[i,j] ,g_y[i,j] ,g_z[i,j]])
                # mo[i,j] = np.linalg.norm([grad_x[i,j] ,grad_y[i,j] ,grad_z[i,j]])
                m[i,j] = -fan[i,j]/mo[i,j] 
        x2, y2, z2 = self.point_gradient_parameter(m*grad_x, m*grad_y, m*grad_z)
        x = x1-x2
        y = y1-y2
        z = z1-z2
        return x , y, z  

    # mu_M
    def mu_M(self, grad_x_01, grad_y_01, grad_z_01, n1_02, n2_02, a_02, b_02, c_02, RotX_1, RotY_1, RotZ_1, RotX_2, RotY_2, RotZ_2):
        n1 = n1_02
        n2 = n2_02
        a = a_02
        b = b_02
        c = c_02
        n = self.n
        grad_inv_x, grad_inv_y, grad_inv_z= np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        mu_x, mu_y, mu_z= np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        grad_x_01_new, grad_y_01_new, grad_z_01_new = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        M1 = ConvertRPYToMat(RotX_1, RotY_1, RotZ_1)
        M2 = ConvertRPYToMat(RotX_2, RotY_2, RotZ_2)
        trans = np.dot(np.transpose(M2), np.transpose(np.linalg.inv(M1)))
        ro, m, fan = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x1 = grad_x_01[i,j]
                x2 = grad_y_01[i,j]
                x3 = grad_z_01[i,j]
                x_arr = np.array([[x1],[x3],[x3]])
                x_trans = np.dot(trans,x_arr) 
                grad_x_01_new[i,j] = x_trans[0]
                grad_y_01_new[i,j] = x_trans[1]
                grad_z_01_new[i,j] = x_trans[2]

                ro[i,j] = np.sign(0.5*a_02*n1_02*grad_x_01_new[i,j]) * np.power(np.abs(0.5*a_02*n1_02*grad_x_01_new[i,j]),2/(2-n2_02)) + \
                          np.sign(0.5*b_02*n1_02*grad_y_01_new[i,j]) * np.power(np.abs(0.5*b_02*n1_02*grad_y_01_new[i,j]),2/(2-n2_02))
                m[i,j] = np.sign(ro[i,j])*np.power(np.abs(ro[i,j]),(n1_02-n2_02)/(4-2*n1_02))
                grad_inv_x[i,j] = np.sign(0.5*a_02*n1_02*grad_x_01_new[i,j])*np.power(np.abs(0.5*a_02*n1_02*grad_x_01_new[i,j]),1/(2-n2_02)) * m[i,j]
                grad_inv_y[i,j] = np.sign(0.5*b_02*n1_02*grad_y_01_new[i,j])*np.power(np.abs(0.5*b_02*n1_02*grad_y_01_new[i,j]),1/(2-n2_02)) * m[i,j]
                grad_inv_z[i,j] = np.sign(0.5*c_02*n1_02*grad_z_01_new[i,j])*np.power(np.abs(0.5*c_02*n1_02*grad_z_01_new[i,j]),1/(2-n1_02))
                # print('内置范数2：',np.linalg.norm([grad_inv_x[i,j] ,grad_inv_z[i,j] ,grad_inv_z[i,j] ]))
                fan[i,j] =  np.power(grad_inv_x[i,j]**2+grad_inv_y[i,j]**2+grad_inv_z[i,j]**2,0.5)
                mu_x[i,j] = grad_inv_x[i,j] / fan[i,j]
                mu_y[i,j] = grad_inv_y[i,j] / fan[i,j]
                mu_z[i,j] = grad_inv_z[i,j] / fan[i,j]
        return mu_x, mu_y, mu_z

    def MinSum_M(self, g_x_M, g_y_M, g_z_M, grad_x, grad_y, grad_z, RotX_1, RotY_1, RotZ_1, RotX_2, RotY_2, RotZ_2):
        M1 = ConvertRPYToMat(RotX_1,RotY_1,RotZ_1)
        M2 = ConvertRPYToMat(RotX_2,RotY_2,RotZ_2)
        # print(M2)
        trans = np.dot(np.transpose(M2), np.transpose(np.linalg.inv(M1)))
        n = self.n
        fan, mo, m = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        grad_x_new, grad_y_new, grad_z_new = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        x, y, z = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        x_new, y_new, z_new = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x = grad_x[i,j]
                y = grad_y[i,j]
                z = grad_z[i,j]
                x_arr = np.array([[x],[y],[z]])
                x_trans = np.dot(trans,x_arr) 
                grad_x_new[i,j] = x_trans[0]
                grad_y_new[i,j] = x_trans[1]
                grad_z_new[i,j] = x_trans[2]
                fan[i,j] = np.power(g_x_M[i,j]**2 + g_y_M[i,j]**2 + g_z_M[i,j]**2, 0.5)
                mo[i,j] = np.power(grad_x_new[i,j]**2 + grad_y_new[i,j]**2 + grad_z_new[i,j]**2, 0.5)
                m[i,j] = -fan[i,j]/mo[i,j] 
        x2, y2, z2 = self.point_gradient_parameter(m*grad_x_new, m*grad_y_new, m*grad_z_new)
        x1, y1, z1 = self.point_gradient_parameter(grad_x, grad_y, grad_z)
        for i in range(n):
            for j in range(n):   
                x2_s= x2[i,j]
                y2_s = y2[i,j]
                z2_s = z2[i,j]
                x2_arr = np.array([[x2_s],[y2_s],[z2_s]])

                x1_s= x1[i,j]
                y1_s = y1[i,j]
                z1_s = z1[i,j]
                x1_arr = np.array([[x1_s],[y1_s],[z1_s]])
                x_sum = np.dot(M1,x1_arr)- np.dot(M2,x2_arr)
                
                x_new[i,j] = x_sum[0] 
                y_new[i,j] = x_sum[1]
                z_new[i,j] = x_sum[2] 
        return x_new , y_new, z_new