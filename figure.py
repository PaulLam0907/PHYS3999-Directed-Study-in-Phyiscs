"""
A module for plotting figures

Package Requirement :
    matplotlib
    PyQt5 (for Python 3.11)
    numpy
"""
import numpy as np
import matplotlib.pyplot as plt


class Figure:
    
    def __init__(self, title = "Figure Title", row = 1, col = 1):
        """
        Create figure object on declaration of this class
        Figure (window) contains axes, each axes contain graphs
        Axes index count from 1 at top left
        
        :param title: str, title of whole figure
        :param row: int
        :param col: int
        """
        self.FIGURE, self.AXES = plt.subplots(nrows = row, ncols = col)  # plt.figure()
        self.FIGURE.suptitle(title)
        self.FIGURE.tight_layout()  # auto adjust axes margin
        
        # flat n-dimensional array to 1D array for convenience
        self.AXES = np.array(self.AXES)
        self.AXES = self.AXES.ravel()
        
    
    def get_all_axes(self):
        return self.AXES
    
    
    def plot(self, x_data, y_data, axes_index = 0, **kwargs):
        """
        Plot the graph on specified axes
        Calling this method for multiple times means plotting multiple curves (graphs) on the same axes
        
        :param x_data: data for x-axis
        :param y_data: data for y-axis
        :param axes_index: int, count from 0, from top left -> top right -> ... -> to bottom right
        :param kwargs: any keyword argument passed to axes.plot()
        :return: None
        """
        axes = self.AXES[axes_index]
        axes.plot(x_data, y_data, **kwargs)
        
        
    def set_axes_properties(self,
                            axes_index = 0,
                            title = None,
                            font_size = 11,
                            xlim = None, ylim = None,
                            xlabel = None, ylabel = None,
                            xgrid = False, ygrid = False,
                            xticks = None, yticks = None,
                            xticklabels = None, yticklabels = None,
                            legend = False):
        """
        Customization for individual axes
        
        :param axes_index: int, count from 0
        :param title: str
        :param font_size: int
        :param xlim: list, [min, max]
        :param ylim: list, [min, max]
        :param xlabel: str, r"..."
        :param ylabel: str, r"..."
        :param xgrid: bool, major grid line for x-axis
        :param ygrid: bool, major grid line for y-axis
        :param xticks: list
        :param yticks: list
        :param xticklabels: list
        :param yticklabels: list
        :param legend: bool
        :return:
        """
        axes = self.AXES[axes_index]
        axes.set_title(title, fontsize = font_size)
        
        if xlim:
            axes.set_xlim(xlim[0], xlim[1])
        
        if ylim:
            axes.set_ylim(ylim[0], ylim[1])
            
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        
        if xgrid:
            axes.xaxis.grid(xgrid)
        
        if ygrid:
            axes.yaxis.grid(ygrid)
        
        if xticks:
            axes.set_xticks(xticks)
            
        if xticklabels:
            axes.set_xticklabels(xticklabels)
        
        if yticks:
            axes.set_yticks(yticks)
        
        if yticklabels:
            axes.set_yticklabels(yticklabels)
        
        if legend:
            axes.legend()
        
        
    def show(self):
        plt.show()
        

# fig = Figure("my fig", 2, 1)
# fig.plot([1, 2, 3], [3, 2, 1], 0)
# fig.plot([3, 2, 1], [3, 2, 1], 1)
# fig.set_axes_properties()
# fig.show()
