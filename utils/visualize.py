import numpy as np
from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self, env='default', **kwargs):
        # self.vis = visdom.Visdom(env=env, **kwargs)

        self.index = {}
        self.log_text = ''

    def has_one_axis(self, X):
        """
        Return True if `X` (tensor or list) has 1 axis
        """
        return (hasattr(X, "ndim") and X.ndim == 1 or
                isinstance(X, list) and not hasattr(X[0], "__len__"))

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """
        Set the axes for matplotlib.
        """
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def plot_with_pyplot(self, X, Y=None,
                         xlabel=None, ylabel=None,
                         legend=None,
                         title=None,
                         xlim=None, ylim=None,
                         xscale='linear', yscale='linear',
                         fmts=('-', 'm--', 'g-.', 'r:'),
                         figsize=(3.5, 2.5),
                         axes=None):
        def set_figsize(figsize=(3.5, 2.5)):
            # display.set_matplotlib_formats('svg')
            plt.rcParams['figure.figsize'] = figsize

        """Plot data points."""
        if legend is None:
            legend = []

        set_figsize(figsize)
        axes = axes if axes else plt.gca()

        if self.has_one_axis(X):
            X = [X]
        if Y is None:
            X, Y = [[]] * len(X), X
        elif self.has_one_axis(Y):
            Y = [Y]
        if len(X) != len(Y):
            X = X * len(Y)
        axes.cla()
        for x, y, fmt in zip(X, Y, fmts):
            if len(x):
                axes.plot(x, y, fmt)
            else:
                axes.plot(y, fmt)

        self.set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        if title:
            plt.title(title)
        plt.show()

    def bbox_to_rect(self, bbox, color):
        """Change (x1, y1, x2, y2) to matplotlib form: ((x, y), width, height)"""
        return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
                             fill=False, edgecolor=color, linewidth=2)

    def show_bboxes(self, axes, bboxes, labels=None, colors=None):
        """Show all bounding boxes"""

        def _make_list(obj, default_values=None):
            if obj is None:
                obj = default_values
            elif not isinstance(obj, (list, tuple)):
                obj = [obj]

            return obj

        labels = _make_list(labels)
        colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])

        for i, bbox in enumerate(bboxes):
            color = colors[i % len(colors)]
            rect = self.bbox_to_rect(np.array(bbox), color)
            axes.add_patch(rect)
            if labels and len(labels) > i:
                text_color = 'k' if color == 'w' else 'w'
                axes.text(rect.xy[0], rect.xy[1], labels[i],
                          va='center', ha='center', fontsize=9, color=text_color,
                          bbox=dict(facecolor=color, lw=0))
