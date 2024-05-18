import numpy as np
from scipy.special import erf
from matplotlib import pyplot as plt

simple_gaussian = lambda x: np.exp(-x**2 / 2)
gaussian = lambda x, x0, w: np.exp(-(x-x0)**2 / w**2)

cfd = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))
simple_skewed = lambda x, a: 2*simple_gaussian(x)*cfd(a*x)
skewed = lambda x, x0, w, a: 2/w * gaussian(x, x0, w) * cfd(a*(x-x0 / w))

x = np.linspace(-3,3,500)
x0 = 0
w = 1
a = 4

y0 = skewed(x,x0,w,0)
y1 = skewed(x,x0,w,1)
y4 = skewed(x,x0,w,4)
yn1 = skewed(x,x0,w,-1)
yn4 = skewed(x,x0,w,-4)
c0 = cfd((x-x0 / w))
c1 = cfd(1*(x-x0 / w))
c4 = cfd(4*(x-x0 / w))
cn1 = cfd(-1*(x-x0 / w))
cn4 = cfd(-4*(x-x0 / w))

plt.plot(x,y0)
plt.plot(x,y1)
plt.plot(x,y4)
plt.plot(x,yn1)
plt.plot(x,yn4)

plt.legend(['a=-4','a=-1','a=0','a=1','a=4'])
plt.show()
