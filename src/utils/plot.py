import numpy as np
import matplotlib.pyplot as plt

def plot_classification(inp, out, model):    
    fig, ax = plt.subplots(1, 1, sharex=True)

    fig.suptitle(model.model_name, fontsize=16)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    labels = np.unique(out)
    colors = ['#dfc27d', '#a6611a', '#f5f5f5', '#80cdc1', '#018571']
    
    delta = 0.025
    border_gap = 0.1
    x = np.arange(min(inp[:, 0]) - border_gap, max(inp[:, 0]) + border_gap, delta)
    y = np.arange(min(inp[:, 1]) - border_gap, max(inp[:, 1]) + border_gap, delta)
    X, Y = np.meshgrid(x, y)
    x_feat = np.reshape(X, X.shape[0] * X.shape[1])
    y_feat = np.reshape(Y, Y.shape[0] * Y.shape[1])
    Z = model.predict(np.array([x_feat, y_feat]).T)
    Z = np.reshape(Z, X.shape)
    
    ax.contourf(X, Y, Z, 20, cmap=plt.cm.BrBG, origin='lower')

    indx_color = 0
    for lbl in labels:
        indx = np.where(out == lbl)
        ax.plot(inp[indx, 0], inp[indx, 1], 'o', color=colors[indx_color])
        indx_color = indx_color + 1
        
    # plt.show()
    plt.savefig('surface.pdf')  


