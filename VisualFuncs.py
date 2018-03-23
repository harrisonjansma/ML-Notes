import matplotlib.pyplot as plt
import numpy as np
from os import system
from sklearn import tree

def Visual_Dec_Regions(list, X, y, clf):
    x_min, x_max = X[:, list[0]].min() - 1, X[:, list[0]].max() + 1
    y_min, y_max = X[:, list[1]].min() - 1, X[:, list[1]].max() + 1
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
