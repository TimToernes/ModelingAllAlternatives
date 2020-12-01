import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import matplotlib.tri as tri


def plot_scatter(hull_sum, inital_solution, data1, data2, weights=[]):
    fig = go.Figure()
    # Hull contour
    fig.add_trace(go.Scatter(x=hull_sum.points[np.append(hull_sum.vertices,hull_sum.vertices[0])][:,0],
                            y=hull_sum.points[np.append(hull_sum.vertices,hull_sum.vertices[0])][:,1],
                            mode='lines'))
    # Initial solution
    #fig.add_trace(go.Scatter(y=[sum(inital_solution[0::2])],
    #                        x=[sum(inital_solution[1::2])],
    #                        mode='markers',name='Initial_solution'))

    fig.add_trace(go.Scatter(x=data1[:,0],
                            y=data1[:,1],
                            #z=data_detail_sum[:,2],
                            mode='markers',name='MGA solutions'))
    if len(weights)>0:
        marker = dict(color=weights,size=10,colorbar={'thickness':20,'title':'Scenario cost'})
    else :
        marker = dict()
    fig.add_trace(go.Scatter(x=data2[:,0],
                            y=data2[:,1],
                            #z=[np.mean(data_detail_sum,axis=0)[2]],
                            mode='markers',name='Interrior_points',
                            marker=marker))
    fig.update_xaxes(title_text='$y_1$')
    fig.update_yaxes(title_text='$y_2$')
    fig.show()
    return 


matplotlib.rcParams.update({'font.size': 18})
def plot_hist(ax,x,w=[] ,color='C1'):
    if len(w)<=0 :
        w = np.ones(len(x))
    kde1 = stats.gaussian_kde(x*1e-3,weights=w)
    xx = np.linspace(7, 43, 1000)
    ax.hist(x=x*1e-3,stacked=False,
                    weights=w,
                    bins=20,
                    density=True,
                    histtype='stepfilled', 
                    alpha=0.5,color=color)
    ax.plot(xx, kde1(xx),color=color)
    return ax



def plot_contour(x,y,z):
    fig, (ax1) = plt.subplots(nrows=1)

    plot_range = ((min(x),max(x)),(min(y),max(y)))

    npts = 200
    ngridx = 1000
    ngridy = 1000
    # -----------------------
    # Interpolation on a grid
    # -----------------------
    # A contour plot of irregularly spaced data coordinates
    # via interpolation on a grid.

    # Create grid values first.
    xi = np.linspace(plot_range[0][0], plot_range[0][1], ngridx)
    yi = np.linspace(plot_range[1][0],plot_range[1][1], ngridy)

    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    #triang = tri.Triangulation(x, y)
    #interpolator = tri.LinearTriInterpolator(triang, z)
    #Xi, Yi = np.meshgrid(xi, yi)
    #zi = interpolator(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    from scipy.interpolate import griddata
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')


    ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr1, ax=ax1)
    #ax1.plot(x, y, 'ko', ms=3)
    ax1.set(xlim=(plot_range[0][0], plot_range[0][1]), 
            ylim=(plot_range[1][0], plot_range[1][1]))
    #ax1.set_title('grid and contour (%d points, %d grid points)' %
    #            (npts, ngridx * ngridy))

    return