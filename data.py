# from nn import nn, scale
import matplotlib.pyplot as plt
import numpy as np, json

np.random.seed(123)

disp = 1/5

def normal(centerx, centery, disp, N):
    x = disp * np.random.randn(N) + centerx
    y = disp * np.random.randn(N) + centery
    return np.column_stack((x, y))

def gen(n, m, N):
    data = np.zeros((1, 3))
    np.random.seed(3)
    for i in range(n):
        for j in range(m):
            cls = np.ones(N) * (1 if ((i + j) % 2 == 0) else 0)
            tmp = normal(i, j, disp, N)
            tmp = np.column_stack((tmp, cls))
            data = np.row_stack((data, tmp))
    return data[1:,:]

def getData():
    N = 15
    square = 2
    ret = gen(square, square, N)
    x, y = ret[:,:2], ret[:,2]
    return x, y


if __name__ == "__main__":
    inp, out = getData()
    plt.scatter(inp[:,0], inp[:,1], c=out, s=50)
    plt.show()
