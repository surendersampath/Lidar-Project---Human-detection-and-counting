import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation






class PolygonHandler:
    # Constructor
    def __init__(self):
        self.X = []
        self.Y = []
        self.numberPoints = 0
        self.fig , ax = plt.subplots()
        self.sc = ax.scatter(self.X,self.Y)
        ax.set_xlim(-2000,10000)
        ax.set_ylim(0,3)

    # Print the polygon
    def update(self,_):
        for i in range(self.numberPoints):
            self.X[i] += np.random.normal()*0.01
            self.Y[i] += np.random.normal()*0.01
        self.sc.set_offsets(np.column_stack((self.X,self.Y)))
        return self.sc,

    # append a point
    def add(self,x,y):
        self.numberPoints += 1
        self.X.append(x)
        self.Y.append(y)

    def autoscale(enable=True, axis='both', tight=None):
        return gca().autoscale(enable=enable, axis=axis, tight=tight)

P = PolygonHandler()


P.add(1,1)
P.add(2,2)
P.add(1,2)

ani = animation.FuncAnimation(P.fig, P.update, interval=10,blit=False)
plt.show()
