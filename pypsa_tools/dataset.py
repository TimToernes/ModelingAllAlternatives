#%%
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
from scipy.spatial import ConvexHull,  Delaunay
from scipy.interpolate import griddata
import sys
import pypsa_tools as pt
from pypsa_tools import *


def rand_split(n):
    
    rand_list = np.random.random(n-1)
    rand_list.sort()
    rand_list = np.concatenate([[0],rand_list,[1]])
    rand_list = np.diff(rand_list)

    return rand_list

class dataset:
    def __init__(self,path,variables=['wind','solar','ocgt'],data_type='excel',m=100000):
        self.m = m #100000
        self.path = path
        self.variables = variables
        self.df_detail = pd.DataFrame()
        self.df_gen_E = pd.DataFrame()
        self.df_store_E = pd.DataFrame()
        self.df_store_p = pd.DataFrame()
        self.df_links = pd.DataFrame()
        self.df_sum_var = pd.DataFrame()
        self.df_secondary_metrics = pd.DataFrame()


        if data_type == 'excel':
            if type(path)==list :
                for p in path:
                    self.load_excel_data(p)
            else:
                self.load_excel_data(path)
        elif data_type == 'csv' :
            if type(path)==list :
                for p in path:
                    self.load_csv_data(p)
            else:
                self.load_csv_data(path)
        else :
            print('unknown data type')


        self.df_sum_var['system_cost'] = self.df_secondary_metrics['system_cost']*1e-6
        #self.variables.append('system_cost')


        self.hull = ConvexHull(self.df_sum_var[self.variables])#,qhull_options='QJ')#,qhull_options='A-0.999')
        #self.hull = ConvexHull(self.df_points.values)#,qhull_options='C-1')#,qhull_options='A-0.999')

        self.create_interior_points()

        
        return 
             

    def load_excel_data(self,path):

        filter = pd.read_excel(path,
                        index_col=0,
                        sheet_name='sum_vars')['wind'] > 0

        self.df_detail = self.df_detail.append(pd.read_excel(path,
                                            index_col=0,
                                            sheet_name='gen_p')[filter],
                                                ignore_index=True)
        self.df_gen_E = self.df_gen_E.append(pd.read_excel(path,
                                                index_col=0,
                                                sheet_name='gen_E')[filter],
                                                    ignore_index=True)
        self.df_store_E = self.df_store_E.append(pd.read_excel(path,
                                                index_col=0,
                                                sheet_name='store_E')[filter],
                                                    ignore_index=True)  
        self.df_store_p = self.df_store_p.append(pd.read_excel(path,
                                                index_col=0,
                                                sheet_name='store_p')[filter],
                                                    ignore_index=True)                                                                                                      
        self.df_sum_var = self.df_sum_var.append(pd.read_excel(path,
                                                index_col=0,
                                                sheet_name='sum_vars')[filter],
                                                    ignore_index=True)
        self.df_secondary_metrics = self.df_secondary_metrics.append(pd.read_excel(path,
                                                index_col=0,
                                                sheet_name='secondary_metrics')[filter],
                                                    ignore_index=True)

        try : 
            self.df_links = self.df_links.append(pd.read_excel(path,
                                        index_col=0,
                                        sheet_name='links')[filter],
                                            ignore_index=True)
        except :
            pass 

        self.df_detail[self.df_detail<0]=0
        self.df_sum_var[self.df_sum_var<0]=0
        return 
    
    def load_csv_data(self,path):

        df = pd.read_csv(path)

        self.df_detail = self.df_detail.append(df.loc[:,df.columns[0:111]],ignore_index=True)
        self.df_gen_E = self.df_gen_E.append(df.loc[:,df.columns[111:222]],ignore_index=True)
        self.df_links = self.df_links.append(df.loc[:,df.columns[222:-4]],ignore_index=True)

        sum_var = dict(wind=np.sum(df.iloc[:,0:111].filter(like='wind'),axis=1),
                        solar=np.sum(df.iloc[:,0:111].filter(like='solar'),axis=1),
                        ocgt =np.sum(df.iloc[:,0:111].filter(like='ocgt'),axis=1),
                        transmission=df['transmission'])
        
        self.df_sum_var = self.df_sum_var.append(pd.DataFrame(sum_var),ignore_index=True)


        secondary_metrics = dict(co2_emission=df['co2_emission'],
                                 system_cost=df['objective_value'],
                                 gini= df['gini']  )

        self.df_secondary_metrics = self.df_secondary_metrics.append(pd.DataFrame(secondary_metrics),ignore_index=True)


    def create_3d_dataset(self):
        type_def = ['ocgt','wind','olar']
        types = [column[-4:] for column in self.df_detail.columns]
        #sort_idx = np.argsort(types)
        idx = [[type_ == type_def[i] for type_ in types] for i in range(len(type_def))]

        points_3D = []
        for row in self.df_detail.iterrows():
            row = np.array(row[1])
            point = [sum(row[idx[0]]),sum(row[idx[1]]),sum(row[idx[2]])]
            points_3D.append(point)

        self.df_points = pd.DataFrame(columns=['ocgt','wind','solar'],data=points_3D)
        #print(self.df_points.head())

        return(self)


    def create_4d_dataset(self):
        transmission = pd.DataFrame({'transmission':self.df_detail['transmission']})

        self.df_points = pd.concat([self.df_points,transmission],axis=1)
        #print(self.df_points.head())
        return self

    def create_generation_data(self):

        type_def = ['ocgt g','wind g','olar g']
        types = [column[-6:] for column in self.df_detail.columns]
        #sort_idx = np.argsort(types)
        idx = [[type_ == type_def[i] for type_ in types] for i in range(len(type_def))]

        points_3D = []
        for row in self.df_detail.iterrows():
            row = np.array(row[1])
            point = [sum(row[idx[0]]),sum(row[idx[1]]),sum(row[idx[2]])]
            points_3D.append(point)

        self.df_generation = pd.DataFrame(columns=['ocgt g','wind g','solar g'],data=points_3D)
        return self

    def create_interior_points(self):
        m = self.m 

        # Generate Delunay triangulation of hull
        try :
            tri = Delaunay(self.df_sum_var[self.variables])#,qhull_options='QJ')#,qhull_options='Qs')#,qhull_options='A-0.999')
        except : 
            points = np.append(self.hull.points[self.hull.vertices],[np.mean(self.hull.points,axis=0)],axis=0)            
            tri = Delaunay(points,qhull_options='Qj')
        # Distribute number of points based on simplex size 
        tri.volumes = []
        for i in range(len(tri.simplices)):
            try :
                tri.volumes.append(ConvexHull(tri.points[tri.simplices[i,:]]).volume)
            except : 
                tri.volumes.append(0)
        tri.volumes = np.array(tri.volumes)
        tri.volumes_norm = tri.volumes/sum(tri.volumes)

        tri.n_points_in_tri =  (tri.volumes_norm*m).astype(int)
        # Generate interrior points of each simplex
        interrior_points = []
        for i in range(len(tri.simplices)):
            tri_face = tri.points[tri.simplices[i,:]]
            for j in range(tri.n_points_in_tri[i]):
                dim = len(tri.points[0,:])
                rand_list = rand_split(dim+1)
                
                new_point = sum([face*rand_n for face,rand_n in zip(tri_face,rand_list)])
                interrior_points.append(new_point)
        
        
        
        self.interrior_points = np.array(interrior_points)
        

        return self

    def calc_interrior_points_cost(self):
        self.interrior_points_cost = griddata(self.df_points.values, 
                                                self.objective_value, 
                                                self.interrior_points,
                                                method='linear')

        self.interrior_points_co2 = griddata(self.df_points.values, 
                                                self.co2_emission, 
                                                self.interrior_points, 
                                                method='linear')
        self.interrior_points_transmission = griddata(self.df_points.values, 
                                                self.transmission, 
                                                self.interrior_points, 
                                                method='linear')
        self.interrior_points_gini = griddata(self.df_points.values, 
                                                self.gini, 
                                                self.interrior_points, 
                                                method='linear')                                                
        return self

    def plot_hull(self):
        #hull = self.hull
        df_points = self.df_points
        #co2_emission = self.co2_emission
        objective_value = self.objective_value
        interrior_points = self.interrior_points

        """
        trace1 = (go.Scatter3d(x=hull.points[hull.vertices][:,0],
                            y=hull.points[hull.vertices][:,1],
                            z=hull.points[hull.vertices][:,2],
                            mode='markers',
                            marker={'color':objective_value,'colorbar':dict(thickness=20)}))
        """
        trace0 = (go.Scatter3d(x=[df_points['ocgt'][0]],
                                    y=[df_points['wind'][0]],
                                    z=[df_points['solar'][0]],
                                    mode='markers',
                                    marker={'color':'blue','size':3}))

        trace1 = (go.Scatter3d(x=df_points['ocgt'][1:],
                                    y=df_points['wind'][1:],
                                    z=df_points['solar'][1:],
                                    mode='markers',
                                    marker={'color':objective_value}))
        # Points generated randomly
        trace2 = (go.Scatter3d(x=interrior_points[:,0],
                                    y=interrior_points[:,1],
                                    z=interrior_points[:,2],
                                    mode='markers',
                                    marker={'size':2,
                                            'color':self.interrior_points_cost,
                                            'colorbar':{'thickness':20,'title':'Scenario cost'}}))

        fig = go.Figure(layout={'width':900,
                                'height':800,
                                'showlegend':False},
                        data=[trace1,trace2,trace0])
        """
        # Plot of facets
        points = hull.points
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            fig.add_trace(go.Mesh3d(x=points[s, 0], 
                                        y=points[s, 1], 
                                        z=points[s, 2],
                                        opacity=1,
                                        color='white'
                                        ))
                                        """
        
        fig.update_layout(scene = dict(
                            xaxis_title='ocgt',
                            yaxis_title='wind',
                            zaxis_title='solar'))
                            
        fig.update_layout(scene_aspectmode='cube')
        return fig

    def plot_histogram(self):
        interrior_points = self.interrior_points*1e-3
        labels = ['ocgt','wind','solar','transmission']
        hist_data = [interrior_points[:,0],interrior_points[:,1],interrior_points[:,2],interrior_points[:,3]]
        fig = ff.create_distplot(hist_data,labels,
                                    bin_size=10000*1e-3,
                                    colors=['#8c564b','#1f77b4','#ff7f0e','#2ca02c'])
        fig.update_xaxes(title_text='MW installed capacity')
        return fig
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
    def plot_cost(self):

        df_points = self.df_points
        #co2_emission = self.co2_emission
        objective_value = self.objective_value
        interrior_points = self.interrior_points
        interrior_points_cost = self.interrior_points_cost
        #interrior_points_co2 = self.interrior_points_co2


        fig = make_subplots(rows = 1, cols=4,shared_yaxes=True)

        fig.add_trace(go.Scatter(y=df_points['ocgt'].values,
                                    x=objective_value,
                                    mode='markers'),row=1,col=1)
        fig.add_trace(go.Scatter(y=interrior_points[:,0],
                                    x=interrior_points_cost,
                                    mode='markers',marker={'opacity':0.1}),row=1,col=1)
        fig.add_trace(go.Scatter(y=df_points['wind'].values,
                                    x=objective_value,
                                    mode='markers'),row=1,col=2)

        fig.add_trace(go.Scatter(y=interrior_points[:,1],
                                    x=interrior_points_cost,
                                    mode='markers',marker={'opacity':0.1}),row=1,col=2)

        fig.add_trace(go.Scatter(y=df_points['solar'].values,
                                    x=objective_value,
                                    mode='markers'),row=1,col=3)

        fig.add_trace(go.Scatter(y=interrior_points[:,2],
                                    x=interrior_points_cost,
                                    mode='markers',marker={'opacity':0.1}),row=1,col=3)
        
        fig.add_trace(go.Scatter(y=self.transmission,
                                    x=objective_value,
                                    mode='markers'),row=1,col=4)

        fig.add_trace(go.Scatter(y=self.interrior_points_transmission,
                                    x=interrior_points_cost,
                                    mode='markers',marker={'opacity':0.1}),row=1,col=4)


        
        fig.update_yaxes(title_text='Installed ocgt [MW]',col=1)
        fig.update_xaxes(title_text='Scenario cost [€]',col=1)
        fig.update_yaxes(title_text=' Installed wind [MW]',col=2)
        fig.update_xaxes(title_text='Scenario cost [€]',col=2)
        fig.update_yaxes(title_text='Installed solar [MW]',col=3)
        fig.update_xaxes(title_text='Scenario cost [€]',col=3)
        fig.update_yaxes(title_text='Installed transmission [MW]',col=4)
        fig.update_xaxes(title_text='Scenario cost [€]',col=4)
        #fig.show()
        return fig





# %%
