#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base PyQt classes for multi-page widgets with change through signal emission.
Page's identifiers are considered to be integers.
Created on Wed Jun 19 11:13:27 2019

@author: nicolas
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QStackedWidget


class PageWidget(QWidget):
    """PyQt widget corresponding to a single page for multi-page widgets."""
    changedPage = pyqtSignal(int)
    
    def emitChangedPage(self, page):
        """Return a function to emit a changedPage signal to the given page."""
        def emit(state):
            self.changedPage.emit(page)
        return emit

class MultiPageWidget(QWidget):
    """Base PyQt widget for multi-pages."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the widget."""
        super().__init__(*args, **kwargs)
        
        # Initialize widget stack
        self.stackedWidget = QStackedWidget(self)
        self.pageWidgets = {}
    
    def addPage(self, pageWidget, pageID, *args, **kwargs):
        """Add the page."""
        pageWidget.changedPage.connect(self.changePage)
        self.stackedWidget.addWidget(pageWidget, *args, **kwargs)
        self.pageWidgets.update({pageID: pageWidget})
    
    def removePage(self, pageID, *args, **kwargs):
        """Remove the page."""
        self.stackedWidget.removeWidget(self.pageWidgets[pageID])
        self.pageWidgets.pop(pageID)
    
    def changePage(self, pageID):
        """Change the current page."""
        self.stackedWidget.setCurrentWidget(self.pageWidgets[pageID])