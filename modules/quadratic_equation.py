import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from termcolor import colored


###############################################
def quadraticEqGenerator(nSamples):
    return np.random.default_rng().uniform(-1,1,(nSamples,3))
###############################################
def quadraticEqSolution(coeff):

    a = coeff[:,0:1]
    b = coeff[:,1:2]
    c = coeff[:,2:3]
    
    delta = b**2 - 4*a*c
    delta = delta.reshape(-1,1)
    
    result = np.where(delta>0, np.sqrt(delta), 0.0)
    result = result*np.array([-1,+1])
    result = (-b+result)/(2*a)
    result = np.where(delta>0, result, (None, None))
    result = np.where(np.abs(a)>1E-10, result, np.array((-c/b, -c/b)).reshape(-1,2))  
    return result 
###############################################
def plotQuadraticEqSolvability(data, interactive=False):

    hasSolution = quadraticEqSolution(data) 
    hasSolution = (hasSolution[:,0]!=None).reshape(-1,1)

    if interactive:
        import plotly.express as px
        nPoints = 500
        fig = px.scatter_3d(x=data[0:nPoints,0], y=data[0:nPoints,1], z=data[0:nPoints,2], 
                        color=hasSolution[0:nPoints,0], size = np.full((nPoints), 0.1),
                        labels={"x":"a", "y":"b", "z":"c"})
        fig.show()
    else:
        fig = plt.figure(figsize=(8,8))
        axis = fig.add_subplot(projection='3d')
        pathes = axis.scatter(data[:,0:1],data[:,1:2], data[:,2:3], c=hasSolution);
        cbar = fig.colorbar(pathes, aspect = 4, fraction=0.10, pad=0.1, 
                            boundaries = (-0.5,0.5,1.5),
                            ticks = (0.0, 1.0),
                            label="has solution?")
        axis.set_xlabel('a')
        axis.set_ylabel('b')
        axis.set_zlabel('c')
###############################################
###############################################
def plotSqEqSolutions(x, y, y_pred):
    
    fig, axes = plt.subplots(1,2, figsize=(12,4))

    pull = (y - y_pred)/y
    pull = pull.flatten()
    threshold = 1E-2
    print(colored("Fraction of events with Y==Y_pred:","blue"),np.mean(np.isclose(y, y_pred)))
    print(colored("Fraction of examples with abs(pull)<0.01:","blue"),"{:3.2f}".format(np.mean(np.abs(pull)<threshold)))
    print(colored("Pull standard deviation:","blue"),"{:3.2f}".format(pull.std()))
    
    axes[0].hist(pull, bins=np.linspace(-1.5,1.5,40), label="(True-Pred)/True");
    axes[0].legend()

    axes[1].axis(False)
    axis = fig.add_subplot(133, projection='3d')

    pull = (y - y_pred)/y
    colors = np.abs(pull)<threshold
    colors = np.sum(colors, axis=1)

    cmapName = plt.rcParams["image.cmap"]
    cmap = mpl.colormaps[cmapName]
    axis.scatter(x[:,0:1], x[:,1:2], x[:,2:3], c = colors);
    axis.scatter((-2), (-2), (-2), label='none correct', marker='o', color=cmap.colors[1])
    axis.scatter((-2), (-2), (-2), label='single correct', marker='o', color=cmap.colors[128])
    axis.scatter((-2), (-2), (-2), label='double correct', marker='o', color=cmap.colors[-1])
    axis.legend(bbox_to_anchor=(1.5,1), loc='upper left')
    axis.set_xlabel("a")
    axis.set_ylabel("b")
    axis.set_zlabel("c");
    axis.set_xlim([-1.1,1.1])
    axis.set_ylim([-1.1,1.1])
    axis.set_zlim([-1.1,1.1])

    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.35, hspace=0.0)
###############################################
###############################################
class QuadraticEquationLoss(tf.keras.losses.Loss):
    
    def __init__(self, name='QuadraticEquationLoss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        a = y_true[:,0:1]
        b = y_true[:,1:2]
        c = y_true[:,2:3]

        loss = a*y_pred**2+b*y_pred+c
        loss = tf.math.reduce_mean(loss**2, axis=1)
        return loss
###############################################
###############################################       