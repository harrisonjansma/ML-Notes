import matplotlib.pyplot as plt
import numpy as np
from os import system
from sklearn import tree
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.colors import ListedColormap

"""
def VDR(cols, X, y, clf):
    x_min, x_max = X[:, cols[0]].min() - 1, X[:, cols[0]].max() + 1
    y_min, y_max = X[:, cols[1]].min() - 1, X[:, cols[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max,0.01))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig_ = plt.figure()
    axis_ = fig_.add_subplot(111)
    axis_.scatter(X[:, cols[0]], X[:, cols[1]], c=y, s=20, edgecolor='k')
    axis_.contourf(xx, yy, Z, alpha=0.4)
    plt.legend(loc = 'lower right')
    plt.show()
"""

def Visual_Dec_Tree(X, trees):
    dotfile = open("dtree2.dot", 'w')
    tree.export_graphviz(trees, out_file = dotfile)
    dotfile.close()
    system('$ dot -Tpng dtree2.dot -o tree.png')


    
def Visual_Logistic( X, y, clf):
    """
    2D Visualization function for logistic regression. Must input a DataFrame and a fitted classifier. Will ouput the decision boundary.
    """
    if type(X)=='pd.core.frame.DataFrame':
        X = X.values
    if type(y)=='pd.core.frame.DataFrame':
        y = y.values 
            
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:,1].reshape(xx.shape)
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                      vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X.iloc[:,0], X.iloc[:, 1], c=y, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
    plt.title('Decision Boundary')
    ax.set(xlabel= X.columns[0], ylabel=X.columns[1])
        
    
    

def VDR(X, y, classifier, resolution=0.02):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'green', 'lightgreen', 'blue', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.6, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   # plot all samples
   X_test, y_test = X[:, :], y[:]
   for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               alpha=0.8, c=cmap(idx),
               marker=markers[idx], label=cl)
   plt.legend(['Died', 'Alive'],loc = 'upper left')


def ThreeD_VDR(X, y, classifier, resolution = 0.02):
    # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'green', 'lightgreen', 'blue', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1


   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   
                          
   np.arange(x2_min, x2_max, resolution))
   pred = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   pred = pred.reshape(xx1.shape)

   fig = plt.figure()
   ax = fig.gca(projection='3d')
   ax.contourf(X[:,0], X[:,1], X[:,2], alpha=0.6, cmap=cmap)
  

   # plot all samples
   X_test, y_test = X[:, :], y[:]
   for idx, cl in enumerate(np.unique(y)):
      ax.scatter(X[y == cl, 0], X[y == cl, 1],X[y == cl, 2],
               alpha=0.8, c=cmap(idx),
               marker=markers[idx], label=cl)
   plt.legend(['Died', 'Alive'],loc = 'upper left')
