import matplotlib.pyplot as plt
import numpy as np

def plot_classification(inp, out, model):    
    fig, ax = plt.subplots(1, 1, sharex=True)

    labels = np.unique(out)
    colors = ['#a6611a', '#80cdc1', '#f5f5f5', '#dfc27d', '#018571']
    
    delta = 0.0025
    border_gap = 0.1
    x = np.arange(min(inp[:, 0]) - border_gap, max(inp[:, 0]) + border_gap, delta)
    y = np.arange(min(inp[:, 1]) - border_gap, max(inp[:, 1]) + border_gap, delta)
    X, Y = np.meshgrid(x, y)
    x_feat = np.reshape(X, X.shape[0] * X.shape[1])
    y_feat = np.reshape(Y, Y.shape[0] * Y.shape[1])
    Z = model.predict(np.array([x_feat, y_feat]).T)
    Z = np.reshape(Z, X.shape)
    
    ax.contour(X, Y, Z, levels=[0.5])

    indx_color = 0
    for lbl in labels:
        indx = np.where(out == lbl)
        ax.plot(inp[indx, 0], inp[indx, 1], 'o', color=colors[indx_color])
        indx_color = indx_color + 1
        
    plt.show()

