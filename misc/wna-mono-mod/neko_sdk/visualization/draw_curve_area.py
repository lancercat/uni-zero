import matplotlib.pyplot as plt
import numpy as np


def add_mean_range_to_figure(ax,xs,mean,upr,lwr,color,label):
    ax.plot(xs, mean, label=label, c=color);
    ax.fill_between(xs, upr, lwr, alpha=0.3, facecolor=color)
    return ax


def add_curve_area(ax,datapack, xrange, yrange):

    mpack=datapack[0]
    color = mpack["color"];
    xs=mpack["x"];
    label=[];
    yss=[];
    for pack in datapack:
        yss.append(pack["y"]);
    yss=np.array(yss);
    mean=yss.mean(axis=0);
    maxv=yss.max(axis=0);
    minv = yss.min(axis=0);
    ax=add_mean_range_to_figure(ax,xs,mean,maxv,minv,color,)
    return ax;
def neko_make_curves_area_chart(ax,datapack):

    color = datapack["color"];
    xs=datapack["x"];
    label=[];
    yss=datapack["y"];
    yss=np.array(yss);
    mean=yss.mean(axis=0);
    maxv=yss.max(axis=0);
    minv = yss.min(axis=0);
    ax=add_mean_range_to_figure(ax,xs,mean,maxv,minv,color,label)
    return ax;


def draw_areas(datapacks,smooth=0):
    fig, ax = plt.subplots();
    for k in datapacks:
        ax=neko_make_curves_area_chart(ax,datapacks[k]);
    plt.show();
    return fig;

if __name__ == '__main__':
    draw()