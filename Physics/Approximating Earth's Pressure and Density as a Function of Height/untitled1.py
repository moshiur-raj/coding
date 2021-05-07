from matplotlib import pyplot as plt
import numpy as np

h = np.arange(0,11,1)
t = np.array([15, 8.5, 2, -4.49, -10.98, -17.47, -23.96, -30.45, -36.94, -43.42, -49.90])
g = np.array([9.807, 9.804, 9.801, 9.797, 9.794, 9.791, 9.788, 9.785, 9.782, 9.779, 9.776])
p = np.array([10.13e4, 8.988e4, 7.950e4, 7.012e4, 6.166e4, 5.405e4, 4.722e4, 4.111e4, 3.565e4, 3.080e4, 2.650e4])
d = np.array([1.225, 1.112, 1.007, 0.9093, 0.8194, 0.7364, 0.6601, 0.5900, 0.5258, 0.4671, 0.4135])
m = d*8.31446261815324*(273.15 + t)/p

#plt.plot(h, t)
#plt.show()
#plt.plot(h, g)
#plt.show()
#plt.plot(h, p)
#plt.show()
#plt.plot(h, d)
#plt.show()
#plt.plot(h, m)
#plt.show()

#m_avg = 0
#for element_of_m in m :
#	m_avg = element_of_m + m_avg

#print(m_avg)

H = []
D = []
P = []
i = float(input())
while i != 'end' :
	H.append(float(i))
	D.append(float(input()))
	P.append(float(input()))
	i = input()

h = h*1000
plt.plot(h, d, 'ro', label = "Experimental Value")
plt.plot(H, D, label = "Prediction")
plt.xlabel("Height (m)")
plt.ylabel("Density (kg/m^3)")
plt.legend()
plt.show()
plt.plot(h, p, 'ro', label = "Experimental Value")
plt.plot(H, P, label = "Prediction")
plt.xlabel("Height (m)")
plt.ylabel("Pressure (Pa)")
plt.legend()
plt.show()