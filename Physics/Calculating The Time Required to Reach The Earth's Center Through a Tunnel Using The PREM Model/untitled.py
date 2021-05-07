from matplotlib import pyplot as plt

r = []
m = []
v = []
g = []

print(input())
i = float(input())
while i != 'end' :
	r.append(float(i))
	m.append(float(input()))
	v.append(float(input()))
	g.append(float(input()))
	i = input()

plt.plot(r, m)
plt.show()
plt.plot(r, v)
plt.show()
plt.plot(r, g)
plt.show()