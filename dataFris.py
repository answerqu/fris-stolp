import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import math
from sklearn import datasets

class Iris:
    def GetData(self):
        iris = datasets.load_iris()
        x = iris.data[:, [2,3]]
        y = iris.target
        return x, y

def Degenerate():
    np.random.seed(123)
    disp = 1/5
    def normal(centerx, centery, disp, N):
        x = disp * np.random.randn(N) + centerx
        y = disp * np.random.randn(N) + centery
        return np.column_stack((x, y))
        
    def getData():
        d1 = 1
        d2 = 1
        np.random.seed(365)
        N = 15
        x1 = normal(-2,0,d1,N)
        y1 = np.array([0] * N)
        x2 = normal(2,0,d2,N)
        y2 = np.array([1] * N)
        x = np.row_stack((x1,x2))
        y = np.concatenate((y1,y2))
        return x, y
        
    return getData()

class DataBuilder:
    def Build(self, name):
        if name == "iris":
            x, y = Iris().GetData()
            return x, y
        elif name == "degenerate":
            x, y = Degenerate()
            print(x)
            return x, y
        else:
            assert("Unknown data")