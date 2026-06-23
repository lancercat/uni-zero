import sys  # We need sys so that we can pass argv to QApplication

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore


# from scipy.interpolate import spline

def neko_make_curves_chart(datapack, xrange, yrange, smooth=False, title="IIIT5k",Yname="Accuracy", *args, **kwargs):
    STY_DICT={"dash":QtCore.Qt.DashLine,
              "solid": QtCore.Qt.SolidLine,
              "dot": QtCore.Qt.DotLine,
              }
    graphWidget = pg.PlotWidget()

    # Add Background colour to white
    graphWidget.setBackground('#ffffff')
    # Add Title
    graphWidget.setTitle(title, color="b", size="15pt")
    # Add Axis Labels
    styles = {"color": "#000000", "font-size": "10pt"}
    graphWidget.setLabel("left", Yname, **styles)
    graphWidget.setLabel("bottom", "Iters (i)", **styles)
    # Add legend
    graphWidget.addLegend(labelTextSize="15pt")
    # Add grid
    graphWidget.showGrid(x=True, y=True)
    # Set Range
    graphWidget.setXRange(xrange[0], xrange[1], padding=0)
    graphWidget.setYRange(yrange[0], yrange[1], padding=0)
    def plot( x, y, plotname, color,style,smooth=False):
        if(smooth):
            x_sm = np.array(x)
            y_sm = np.array(y)
            x_smooth = np.linspace(x_sm.min(), x_sm.max(), smooth)
            y_smooth = spline(x, y, x_smooth)
        else:
            x_smooth=x;
            y_smooth=y;
        pen = pg.mkPen(color=color,width=4,style=style);
        graphWidget.plot(x_smooth, y_smooth, name=plotname, pen=pen,symbol=pack["s"])

    for pack in datapack:
        plot(pack["x"], pack["y"], pack["n"], pack["c"], STY_DICT[pack["l"]], smooth=smooth);
    return graphWidget;


class MainWindow(QtWidgets.QMainWindow):
    def __init__(this,datapack,xrange,yrange,smooth=False, *args, **kwargs):
        super(MainWindow, this).__init__(*args, **kwargs)
        this.graphWidget=neko_make_curves_chart(datapack, xrange, yrange, smooth,"magic", args, kwargs)
        this.setCentralWidget(this.graphWidget);

def draw(xrange,yrange,datapack,smooth=0):
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(datapack,xrange,yrange,smooth)
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    draw()