import matplotlib.pyplot as plt
import numpy as np


###################################################################### 
######################################################################
def plotTrainHistory(history):
    
    fig, axes = plt.subplots(1,2, figsize=(7,3))
    axes[0].plot(history.history['loss'], label = 'train')
    axes[0].plot(history.history['val_loss'], label = 'validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss function')
    axes[0].legend(loc='upper right')
    
    axes[1].plot(history.history['loss'], label = 'train')
    axes[1].plot(history.history['val_loss'], label = 'validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss function')
    axes[1].legend(loc='upper right')
    axes[1].set_yscale('log')
    
    plt.subplots_adjust(bottom=0.02, left=0.02, right=0.98, wspace=0.6)
    plt.savefig("fig_png/training_history.png", bbox_inches="tight")
###################################################################### 
###################################################################### 
def plot_PDF_CDF(z):
    fig, axes = plt.subplots(1,2, figsize=(12, 6))
    
    ##########
    axes[0].hist(z, bins=50, density=True, label="PDF")
    x = np.linspace(-3,3, 100)
    y = 1.0/np.sqrt(2*np.pi)*np.exp(-x**2/2)
    axes[0].plot(x,y, 'r', label="N(0,1)")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("p(z)")
    axes[0].legend(loc="upper right");
    
    ##########
    axes[1].hist(z, color='sandybrown', bins=50, cumulative=True, weights = np.full_like(z, 1/z.size), label="CDF")
    axes[1].plot((0,0), (0,1), color='black')
    axes[1].plot((-3, 3), (0.5, 0.5), color='black')
    axes[1].set_xlabel("z")
    axes[1].set_ylabel(r"$\int_{-\infty}^{z} p(z) dz$")
    axes[1].legend(loc="upper left");
    ##########
    
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.3)
    plt.savefig("fig_png/example1.png", bbox_inches="tight")
###################################################################### 
###################################################################### 

