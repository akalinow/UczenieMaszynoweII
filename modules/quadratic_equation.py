import numpy as np
import matplotlib.pyplot as plt

###############################################
def quadraticEqGenertor(nSamples):
    return np.random.default_rng().uniform(-1,1,(nSamples,3))
###############################################
def quadraticEqSolution(coeff):

    a = coeff[:,0:1]
    b = coeff[:,1:2]
    c = coeff[:,2:3]
    
    delta = b**2 - 4*a*c
    delta = delta.reshape(-1,1)
    
    result = np.where(delta>0, np.sqrt(delta), 0.0)
    result = result*np.array([-1,1])
    result = (result - b)/(2.0*a)
    result = np.where(delta>0, result, None)
    
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