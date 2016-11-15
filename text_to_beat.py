import matplotlib.pyplot as plt
f = open('open_0' + str(19) + '.txt')
c = f.read().split()
a = [float(x) for x in c]
b = []
for i in range(1,len(a)-1):
   b.append(60/(a[i] - a[i-1]))
plt.plot(b)
plt.show()
print sum(b)/len(b)
print max(b) - min(b)
