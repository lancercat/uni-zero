
import sys;

import pyqtgraph
import pyqtgraph as pg;
from PySide6 import QtCore, QtWidgets


class neko_qt_chart_gui:
    @classmethod
    def show(cls, widget, path=None,title="meow"):
        if not QtWidgets.QApplication.instance():
            app = QtWidgets.QApplication(sys.argv)
        else:
            app = QtWidgets.QApplication.instance()
        plot = pg.PlotWidget()
        mw = QtWidgets.QMainWindow()
        mw.setWindowTitle(title)
        if path is None:
            mw.resize(800, 800)
        else:
            mw.resize(8000, 8000);

        cw = QtWidgets.QWidget()
        mw.setCentralWidget(cw)
        l = QtWidgets.QVBoxLayout()
        cw.setLayout(l)
        # plot.addItem(w1);
        plot.addItem(widget);
        l.addWidget(plot)
        if (path is not None):
            # mw.show();
            pix = mw.grab()
            pix.save(path)
        else:
            mw.show();
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                QtWidgets.QApplication.instance().exec_();

    @classmethod
    def draw_multi(this, widgets, path=None):
        cw = QtWidgets.QWidget()
        if path is None:
            pass
            # mw.resize(800, 800)
        else:
            cw.resize(8000, 4500);

        l = QtWidgets.QGridLayout()
        cw.setLayout(l)
        i = 0;
        c = 3
        # plot.addItem(w1);
        for w in widgets:
            if (type(w) != pyqtgraph.PlotWidget):
                pgw=pyqtgraph.PlotWidget();
                pgw.addItem(w);
            else:
                pgw=w;
            l.addWidget(pgw, i % c, i // c);
            i += 1;
        return cw;
    @classmethod
    def show_multi(this, widgets, path=None):
        if not QtWidgets.QApplication.instance():
            app = QtWidgets.QApplication(sys.argv)
        else:
            app = QtWidgets.QApplication.instance()
        mw = QtWidgets.QMainWindow()
        mw.setWindowTitle("meow")
        if path is None:
            mw.resize(800, 800)
        else:
            mw.resize(8000, 8000);

        cw = this.draw_multi(widgets,path);
        mw.setCentralWidget(cw)
        if (path is not None):
            # mw.show();
            pix = mw.grab()
            pix.save(path)
        else:
            mw.show();
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                QtWidgets.QApplication.instance().exec_();
