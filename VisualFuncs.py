import matplotlib.pyplot as plt
import numpy as np
from os import system
from sklearn import tree
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def VDR(cols, X, y, clf):
    x_min, x_max = X[:, cols[0]].min() - 1, X[:, cols[0]].max() + 1
    y_min, y_max = X[:, cols[1]].min() - 1, X[:, cols[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig_ = plt.figure()
    axis_ = fig_.add_subplot(111)
    axis_.scatter(X[:, list[0]], X[:, list[1]], c=y, s=20, edgecolor='k')
    axis_.contourf(xx, yy, Z, alpha=0.4)
    plt.show()

def Visual_Dec_Tree(X, trees):
    dotfile = open("dtree2.dot", 'w')
    tree.export_graphviz(trees, out_file = dotfile)
    dotfile.close()
    system('$ dot -Tpng dtree2.dot -o tree.png')

def ThreeD_VDR(cols, X, target, clf, Variable1 = 'Variable 1', Variable2 = 'Variable 2'):
    x_min, x_max = X[:, cols[0]].min() - 1, X[:, cols[0]].max() + 1
    y_min, y_max = X[:, cols[1]].min() - 1, X[:, cols[1]].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    onlyX = pd.DataFrame({Variable1: xx.ravel(), Variable2 : yy.ravel()})
    
    predicted = clf.predict(onlyX)
    
    fig = plt.figure( figsize = (6,6))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X[:,cols[0]], X[:,cols[1]], target, c = 'blue', marker = 'o', alpha = 0.4)
    ax.plot_surface(xx, yy, predicted.reshape(xx.shape))
    plt.xlabel(Variable1)
    plt.ylabel(Variable2)
    plt.show()
    
    
    