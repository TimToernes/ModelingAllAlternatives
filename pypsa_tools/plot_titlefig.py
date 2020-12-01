import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial import ConvexHull,  Delaunay



def plot_titlefig():
    points = [[0.54118384, 0.94067655, 0.54699987],
        [0.73502797, 0.16935739, 0.55790591],
        [0.96576643, 0.29232711, 0.20454971],
        [0.55863702, 0.1188831 , 0.65077112],
        [0.38291315, 0.67108752, 0.08551867],
        [0.14490251, 0.56089522, 0.7446059 ],
        [0.43321413, 0.29950656, 0.78427763],
        [0.23649422, 0.61364544, 0.31589086],
        [0.40290789, 0.76772521, 0.71956596],
        [0.48181454, 0.10971972, 0.78060066],
        [0.22610062, 0.50378525, 0.88588253],
        [0.75489242, 0.61593716, 0.15210322],
        [0.73144735, 0.52480853, 0.73914978],
        [0.83217595, 0.77561736, 0.79231154],
        [0.47590843, 0.50194603, 0.47847918],
        [0.59244176, 0.09104889, 0.56839673],
        [0.30216797, 0.85553799, 0.64035338],
        [0.49417155, 0.56424763, 0.79651933],
        [0.35106255, 0.53388753, 0.91428053],
        [0.18259525, 0.25313506, 0.72112773],
        [0.1802845 , 0.61119022, 0.94388547],
        [0.00801884, 0.1606304 , 0.23368452],
        [0.30003447, 0.04324177, 0.32690292],
        [0.43139553, 0.93269856, 0.83140476],
        [0.00530553, 0.8967605 , 0.99334681],
        [0.73616582, 0.9718411 , 0.4826711 ],
        [0.1317993 , 0.22366495, 0.17890496],
        [0.47125957, 0.38791989, 0.31627866],
        [0.43600545, 0.81458297, 0.12069482],
        [0.63466639, 0.33219955, 0.18803344]]

    hull = ConvexHull(points)


    """ Colors
            '#1f77b4',  // muted blue
            '#ff7f0e',  // safety orange
            '#2ca02c',  // cooked asparagus green
            '#d62728',  // brick red
            '#9467bd',  // muted purple
            '#8c564b',  // chestnut brown
            '#e377c2',  // raspberry yogurt pink
            '#7f7f7f',  // middle gray
            '#bcbd22',  // curry yellow-green
            '#17becf'   // blue-teal
        """

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=hull.points[hull.vertices][:,0],
                        y=hull.points[hull.vertices][:,1],
                        z=hull.points[hull.vertices][:,2],
                        mode='markers',
                        marker={'color':'#17becf','size':4}))


    # Plot of facets
    h_points = hull.points
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        fig.add_trace(go.Mesh3d(x=h_points[s, 0], 
                                    y=h_points[s, 1], 
                                    z=h_points[s, 2],
                                    opacity=0.7,
                                    color='pink'
                                    ))
        fig.add_trace(go.Scatter3d(x=h_points[s, 0], 
                                    y=h_points[s, 1], 
                                    z=h_points[s, 2],
                                    opacity=0.7,
                                    mode='lines',
                                    line=dict(width=5,color='white'),
                                    #color='black'
                                    ))                                

    fig.update_layout(showlegend=False,
                    scene = dict(
                        xaxis=dict(
                            showbackground=False,
                            showaxeslabels=False,
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            title='',
                            visible=False,

                        ),
                        yaxis=dict(
                            showbackground=False,
                            showaxeslabels=False,
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            title='',
                            visible=False,
                        ),
                        zaxis=dict(
                            showbackground=False,
                            showaxeslabels=False,
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            title='',
                            visible=False,
                        ),
                        ))
                        
    fig.update_layout(scene_aspectmode='cube')
    fig.update_layout(
        autosize=False,
        width=900,
        height=900,)

    #fig.write_image(im_dir+"title_pic.pdf")
    return fig
 