#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the frame-wise correction window widget.
The user can modify the results on a frame by frame basis on this page, e.g. by 
changing the identities of ROIs, and/or correcting the segmentations.
Created on Wed Jun 19 11:48:44 2019

@author: nicolas
"""

import os.path
import warnings
import shutil
import pickle

import numpy as np
from matplotlib import cm, colorbar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage import io, measure
from PyQt5.QtCore import Qt, pyqtSignal, QStringListModel
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (QAbstractItemView, QButtonGroup, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, 
                             QLabel, QLineEdit, QListView, QMessageBox, QProgressDialog, QPushButton, 
                             QSizePolicy, QVBoxLayout)
import cv2

from .constants import PAGE_MODEL, CHOICE_PATHS, ID_CMAP, BIN_S, RATE_HZ
from .image import array2pixmap, LabelImage, LabelStack, VerticalScrollImage
from .multipage import AxoidPage
from axoid.utils.image import imread_to_float, to_id_cmap, overlay_mask, overlay_mask_stack
from axoid.utils.fluorescence import get_fluorophores, compute_fluorescence, save_fluorescence


## Constants
# Drawing modes
IDLE = 0
SETID = 1
DISCARDING = 2


class CorrectionPage(AxoidPage):
    """
    Page of the frame-wise correction process.
    
    Here, the user can modify the results on a frame by frame basis, e.g. by 
    changing the identities of ROIs, and/or correcting the segmentation.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the correction page.
        
        Args and kwargs are passed to the AxoidPage constructor.
        """
        super().__init__(*args, **kwargs)
    
    def initUI(self):
        """Initialize the interface."""
        # Lists of frames with pending modifications, and with applied modifications
        self.edited_frames = QStringListModel([])
        self.applied_frames = QStringListModel([])
        
        # Create intermediate gui folder
        for folder in ["axoid_internal", "GC6_auto", "ROI_auto"]:
            gui_path = os.path.join(self.experiment, "output", folder, "gui")
            if os.path.isdir(gui_path):
                shutil.rmtree(gui_path)
            shutil.copytree(os.path.join(self.experiment, "output", folder, "final"),
                            gui_path)
        
        # Initialize images and stacks
        model_path = os.path.join(self.experiment, "output", "axoid_internal",
                                  "final", "model.tif")
        model_img = io.imread(model_path)
        self.n_axons_max = model_img.max()
        model_img = to_id_cmap(model_img, cmap=ID_CMAP)
        self.model = LabelImage(model_img)
        
        # EditChoice is the middle editable image which the user can choose
        self.editChoice_layouts = dict()
        self.editChoice_widgets = dict()
        self.segmentations = {
            "final": LabelStack(os.path.join(self.experiment, "output", 
                         "axoid_internal", "final", "segmentations.tif")),
            "gui": SegmentationStack(os.path.join(self.experiment, "output", 
                       "axoid_internal", "gui", "segmentations.tif"))
        }
        identities = io.imread(os.path.join(self.experiment, "output", 
                         "axoid_internal", "final", "identities.tif"))
        self.identities = {
            "final": LabelStack(to_id_cmap(identities, cmap=ID_CMAP)),
            "gui": IdentityStack(os.path.join(self.experiment, "output", 
                       "axoid_internal", "gui", "identities.tif"),
                       seg_stack=os.path.join(self.experiment, "output", 
                       "axoid_internal", "gui", "segmentations.tif"))
        }
        self.identities["gui"].editedFrame[int].connect(self.addEditedFrames)
        self.rois = {
            "final": LabelStack(os.path.join(self.experiment, "output", 
                         "ROI_auto", "final", "RGB_seg.tif")),
            "gui": ROIStack(os.path.join(self.experiment, "output", "ROI_auto",
                                         "gui", "RGB_seg.tif"))
        }
            
        for folder in ["final", "gui"]:
            self.editChoice_layouts.update({folder: QHBoxLayout()})
            self.editChoice_widgets.update({folder: QLabel()})
            self.editChoice_layouts[folder].addWidget(
                    self.editChoice_widgets[folder], alignment=Qt.AlignCenter)
            
            self.choice_layouts.update({folder: QHBoxLayout()})
            self.choice_widgets.update({folder: QLabel()})
            self.choice_layouts[folder].addWidget(self.choice_widgets[folder],
                                                  alignment=Qt.AlignCenter)
        
        ## Left display
        for row in [1, 2]:
            self.left_display.setRowStretch(row, 1)
        for col in [1, 2, 3]:
            self.left_display.setColumnStretch(col, 1)
        
        # Row and column labels
        labels = ["Model", "In folder", "Current"]
        positions = [(0,1), (1,0), (2,0)]
        alignments = [Qt.AlignCenter] + [Qt.AlignVCenter | Qt.AlignLeft] * 2
        for label, position, alignment in zip(labels, positions, alignments):
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignCenter)
            self.left_display.addWidget(lbl, *position, alignment=alignment)
        
        # Model image
        self.left_display.addWidget(self.model, 1, 1, 2, 1, alignment=Qt.AlignCenter)
        
        # Dropdown choice for editable (middle) column
        self.combo_editChoice = QComboBox()
        self.combo_editChoice.setFocusPolicy(Qt.ClickFocus)
        self.combo_editChoice.addItems(["identities"]) # "segmentations" & "ROI" not implemented
        self.combo_editChoice.setCurrentText("identities")
        self.combo_editChoice.activated[str].connect(self.changeEditChoice)
        self.left_display.addWidget(self.combo_editChoice, 0, 2, alignment=Qt.AlignCenter)
        # Middle column images
        for folder, position in zip(["final", "gui"], [(1,2), (2,2)]):
            self.left_display.addLayout(self.editChoice_layouts[folder], *position)
        
        # Dropdown choice for last column
        self.combo_choice = QComboBox()
        self.combo_choice.setFocusPolicy(Qt.ClickFocus)
        self.combo_choice.addItems(["input", "rgb_init", "seg_init", "ROI", "ΔR/R", "ΔF/F"])
        self.combo_choice.setCurrentText("ROI")
        self.changeChoice(self.combo_choice.currentText())
        self.combo_choice.activated[str].connect(self.changeChoice)
        self.left_display.addWidget(self.combo_choice, 0, 3, alignment=Qt.AlignCenter)
        # Last column images
        for folder, position in zip(["final", "gui"], [(1,3), (2,3)]):
            self.left_display.addLayout(self.choice_layouts[folder], *position)
        
        ## Right control with buttons and actions
        # Get colorbar image
        fig, ax = plt.subplots(1, 1, figsize=(0.5,3),
                               facecolor=[1,1,1,0], constrained_layout=True)
        cb = colorbar.ColorbarBase(ax, cmap=cm.get_cmap(ID_CMAP))
        cb.ax.invert_yaxis()
        cb.set_ticks(np.arange(self.n_axons_max) / (self.n_axons_max - 1))
        cb.set_ticklabels(np.arange(self.n_axons_max))
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        canvas_hbox = QHBoxLayout()
        canvas_hbox.addStretch(1)
        canvas_hbox.addWidget(self.canvas)
        canvas_hbox.addStretch(1)
        
        # Toolboxes
        self.tool_actions = QVBoxLayout()
        self.tool_actions_widget = QLabel()
        self.tool_actions.addWidget(self.tool_actions_widget)
        self.tool_actions.addStretch(1)
        # Update edit choice combobox now that toolboxes exists
        self.changeEditChoice(self.combo_editChoice.currentText())
        
        # List of edited frames
        edited_frames_layout = QGridLayout()
        edited_frames_layout.setRowStretch(1, 1)
        edited_frames_layout.addWidget(QLabel("Edited frames"), 0, 0, 
                                       alignment=Qt.AlignBottom | Qt.AlignVCenter)
        edited_frames_layout.addWidget(QLabel("Applied frames"), 0, 1, 
                                       alignment=Qt.AlignBottom | Qt.AlignVCenter)
        edited_list = QListView()
        edited_list.setModel(self.edited_frames)
        edited_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        edited_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        edited_frames_layout.addWidget(edited_list, 1, 0, 
                                       alignment=Qt.AlignTop | Qt.AlignVCenter)
        applied_list = QListView()
        applied_list.setModel(self.applied_frames)
        applied_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        applied_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        edited_frames_layout.addWidget(applied_list, 1, 1, 
                                       alignment=Qt.AlignTop | Qt.AlignVCenter)
        
        # Apply/save/etc.
        fn_vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip("Apply edited frames to current results")
        apply_btn.clicked.connect(self.applyChanges)
        save_btn = QPushButton("Save")
        save_btn.setToolTip("Save current results in the final/ folder")
        save_btn.clicked.connect(self.saveChanges)
        hbox.addWidget(apply_btn)
        hbox.addWidget(save_btn)
        finish_btn = QPushButton("Finish")
        finish_btn.setToolTip("Finish AxoID's user correction GUI with current results")
        finish_btn.clicked.connect(self.finish)
        annotation_btn = QPushButton("Manual annotation")
        annotation_btn.setToolTip("Discard all results and move on to manual annotation (not implemented)")
        annotation_btn.setEnabled(False)
        model_btn = QPushButton("Back to model")
        model_btn.setToolTip("Return to the model correction page")
        model_btn.clicked.connect(self.gobackModel)
        
        fn_vbox.addLayout(hbox)
        fn_vbox.addWidget(finish_btn)
        fn_vbox.addWidget(model_btn)
        fn_vbox.addStretch(1)
        fn_vbox.addWidget(annotation_btn)
        
        self.right_title.setText("<b>Frame correction</b>")
        self.right_control.addWidget(QLabel("Axon identity"), alignment=Qt.AlignCenter)
        self.right_control.addLayout(canvas_hbox)
        self.right_control.addStretch(1)
        self.right_control.addLayout(self.tool_actions, stretch=10)
        self.right_control.addLayout(edited_frames_layout, stretch=10)
        self.right_control.addStretch(1)
        self.right_control.addLayout(fn_vbox, stretch=6)
    
    def changeFrame(self, num):
        """
        Change the stack frames to display.
        
        Parameters
        ----------
        num : int
            Index of the frame in the stacks to display.
        """
        super().changeFrame(num)
        for folder in ["final", "gui"]:
            if isinstance(self.editChoice_widgets[folder], LabelImage):
                self.editChoice_widgets[folder].changeFrame(int(num))
            if isinstance(self.choice_widgets[folder], LabelImage):
                self.choice_widgets[folder].changeFrame(int(num))
    
    def changeEditChoice(self, choice):
        """
        Change the editable choice display based on the ComboBox choice.
        
        Parameters
        ----------
        choice : str
            String corresponding to the user choice. See constants.py for a list
            of possible strings, and their corresponding files.
        """
        if choice == "segmentation":
            new_editChoice = self.segmentations
        elif choice == "identities":
            new_editChoice = self.identities
        elif choice == "ROI":
            new_editChoice = self.rois
        for folder in ["final", "gui"]:
            self.editChoice_layouts[folder].removeWidget(self.editChoice_widgets[folder])
            self.editChoice_widgets[folder].hide()
            self.editChoice_widgets[folder] = new_editChoice[folder]
            self.editChoice_widgets[folder].show()
            self.editChoice_layouts[folder].addWidget(
                    self.editChoice_widgets[folder], alignment=Qt.AlignCenter)
        
        # Reset tools
        self.identities["gui"].change_mode(IDLE)
        
        # Change the toolbox
        tool_groupbox = QGroupBox("Toolbox: " + choice)
        tool_groupbox.setAlignment(Qt.AlignCenter)
        self.tool_group = QButtonGroup()
        tool_layout = QVBoxLayout()
        
        general_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset the frame to the one in the final/ folder")
        undo_btn = QPushButton("Undo")
        undo_btn.setToolTip("Undo the last change on the frame")
        undo_btn.setShortcut("Ctrl+Z")
        general_layout.addWidget(reset_btn)
        general_layout.addWidget(undo_btn)
        tool_layout.addLayout(general_layout)
        if choice == "segmentation":
            pass
        elif choice == "identities":
            reset_btn.clicked.connect(self.identities["gui"].reset)
            undo_btn.clicked.connect(self.identities["gui"].undo_last)
            
            setID_hbox = QHBoxLayout()
            setID_btn = QPushButton("Set ID:")
            setID_btn.setToolTip("Click on an axon to set its identity")
            setID_btn.setCheckable(True)
            setID_btn.clicked[bool].connect(self.enableSetID)
            setID_combox = QComboBox()
            setID_combox.setFocusPolicy(Qt.ClickFocus)
            setID_combox.addItems([str(i) for i in range(self.n_axons_max)])
            setID_combox.activated[str].connect(self.identities["gui"].set_set_id)
            setID_hbox.addWidget(setID_btn)
            setID_hbox.addWidget(setID_combox)
            discard_btn = QPushButton("Discard")
            discard_btn.setToolTip("Click on an axon to set its identity to 0 (background)")
            discard_btn.setCheckable(True)
            discard_btn.clicked[bool].connect(self.enableDiscard)
            
            tool_layout.addLayout(setID_hbox)
            tool_layout.addWidget(discard_btn)
            self.tool_group.addButton(setID_btn)
            self.tool_group.addButton(discard_btn)
        elif choice == "ROI":
            erase_hbox = QHBoxLayout()
            erase_btn = QPushButton("Erase ROI:")
            erase_btn.setToolTip("Erase the ROI corresponding to the identity")
            erase_combox = QComboBox()
            erase_combox.setFocusPolicy(Qt.ClickFocus)
            erase_combox.addItems([str(i) for i in range(self.n_axons_max)])
            erase_hbox.addWidget(erase_btn)
            erase_hbox.addWidget(erase_combox)
            add_hbox = QHBoxLayout()
            add_btn = QPushButton("Add ROI:")
            add_btn.setToolTip("Draw an elliptic ROI on the frame, centered on the mouse.\n"
                               "W is width, H is height and θ orientation.")
            add_btn.setCheckable(True)
            add_combox = QComboBox()
            add_combox.setFocusPolicy(Qt.ClickFocus)
            add_combox.addItems([str(i) for i in range(self.n_axons_max)])
            add_hbox.addWidget(add_btn)
            add_hbox.addWidget(add_combox)
            add_hbox2 = QHBoxLayout()
            def get_add_edit():
                """Return a QLineEdit of length 3 for numbers-"""
                edit = QLineEdit()
                edit.setMaxLength(3)
                edit.setMaximumWidth(30)
                edit.setValidator(QIntValidator(0, 999))
                return edit
            add_hbox2.addWidget(QLabel("W:"))
            add_edit_W = get_add_edit()
            add_hbox2.addWidget(add_edit_W)
            add_hbox2.addWidget(QLabel("H:"))
            add_edit_H = get_add_edit()
            add_hbox2.addWidget(add_edit_H)
            add_hbox2.addWidget(QLabel("θ:"))
            add_edit_T = get_add_edit()
            add_hbox2.addWidget(add_edit_T)
            
            tool_layout.addLayout(erase_hbox)
            tool_layout.addLayout(add_hbox)
            tool_layout.addLayout(add_hbox2)
            self.tool_group.addButton(add_btn)
        tool_groupbox.setLayout(tool_layout)
        self.tool_actions.replaceWidget(self.tool_actions_widget, tool_groupbox)
        self.tool_actions_widget.close()
        self.tool_actions_widget = tool_groupbox
    
    def changeChoice(self, choice):
        """
        Change the choice display based on the ComboBox choice.
        
        Parameters
        ----------
        choice : str
            String corresponding to the user choice. See constants.py for a list
            of possible strings, and their corresponding files.
        """
        for folder in ["final", "gui"]:
            self.choice_layouts[folder].removeWidget(self.choice_widgets[folder])
            self.choice_widgets[folder].close()
            
            path = os.path.join(self.experiment, CHOICE_PATHS[choice] % folder)
            image = io.imread(path)
            
            if choice in ["ΔR/R", "ΔF/F"]:
                self.choice_widgets[folder] = VerticalScrollImage(image)
            elif image.ndim == 2:
                self.choice_widgets[folder] = LabelImage(image)
            elif image.ndim == 3:
                if image.shape[-1] in [3, 4]: # assume RGB(A)
                    self.choice_widgets[folder] = LabelImage(image)
                else:
                    self.choice_widgets[folder] = LabelStack(image)
                    self.choice_widgets[folder].changeFrame(self.nframe_sld.value())
            else:
                    self.choice_widgets[folder] = LabelStack(image)
                    self.choice_widgets[folder].changeFrame(self.nframe_sld.value())
            
            
            self.choice_layouts[folder].addWidget(self.choice_widgets[folder],
                                                  alignment=Qt.AlignCenter)
    
    def enableSetID(self, checked):
        """Enable the fusing tool."""
        if checked:
            self.identities["gui"].change_mode(SETID)
    
    def enableDiscard(self, checked):
        """Enable the discarding tool."""
        if checked:
            self.identities["gui"].change_mode(DISCARDING)
    
    def addEditedFrames(self, index):
        """
        Add the given index to the list of edited frames.
        
        Parameters
        ----------
        index : int
        print(len(self.changes[self.index]))  
            Index of the frames that has been edited.
        """
        # Verify that index is not already there
        if str(index) in self.edited_frames.stringList():
            return
        string_list = self.edited_frames.stringList()
        string_list.append(str(index))
        string_list.sort(key=lambda x: int(x))
        self.edited_frames.setStringList(string_list)
    
    def _get_progress_dialog(self, text, vmin=0, vmax=100):
        """
        Return a QProgressDialog and start to display it.
        
        It only shows the text with the progress bar, and there are no cancel
        button. It starts from vmin, and disappears after reaching vmax.
        To update its value, use progress.setValue(val)
        """
        progress = QProgressDialog(text, "", vmin, vmax)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        return progress
    
    def applyChanges(self):
        """Apply the modified frames to the output."""
        if self.edited_frames.rowCount() == 0:
            return
        progress = self._get_progress_dialog("Applying changes...")
                
        # Save identities
        path = os.path.join(self.experiment, "output", "axoid_internal",
                            "gui", "identities.tif")
        self.identities["gui"].applyEditions()
        identities = self.identities["gui"].identities
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Warning)
            io.imsave(path, identities)
        progress.setValue(33)
        
        ## ROI_auto (raw data + contours)
        path = os.path.join(self.experiment, "output", "axoid_internal",
                            "gui", "input.tif")
        input_data = io.imread(path)
        path = os.path.join(self.experiment, "output", "ROI_auto",
                            "gui", "RGB_seg.tif")
        input_roi = io.imread(path)
        ids = np.unique(identities)
        ids = ids[ids != 0]
        with open(os.path.join(self.experiment, "output", "GC6_auto", "gui", 
                               "All_contour_list_dic_Usr.p"), "rb") as f:
            contour_list = pickle.load(f)["AllContours"]
        edited_str_list = self.edited_frames.stringList()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Warning)
            for i in range(len(input_roi)):
                if str(i) not in edited_str_list:
                    continue
                contour_list[i].clear()
                input_roi[i] = input_data[i].copy()
                for n, id in enumerate(ids):
                    roi_img = (identities[i] == id)
                    if np.sum(roi_img) == 0:
                        contour_list[i].append([])
                        continue
                    _, contours, _ = cv2.findContours(roi_img.astype(np.uint8),
                                                      cv2.RETR_LIST,
                                                      cv2.CHAIN_APPROX_NONE)
                    x, y = contours[0][:,0,0].max(), contours[0][:,0,1].min() # top-right corner of bbox
                    # Draw the contour and write the ROI id
                    cv2.drawContours(input_roi[i], contours, -1, (255,255,255), 1)
                    cv2.putText(input_roi[i], str(n), (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                    contour_list[i].append(contours[0])
        # Save results
        path = os.path.join(self.experiment, "output", "ROI_auto",
                            "gui", "RGB_seg.tif")
        io.imsave(path, input_roi)
        with open(os.path.join(self.experiment, "output", "GC6_auto", "gui", 
                               "All_contour_list_dic_Usr.p"), "wb") as f:
            pickle.dump({"AllContours": contour_list}, f)
        progress.setValue(66)
        
        ## Fluorescence traces
        len_baseline = int(BIN_S * RATE_HZ + 0.5) # number of frames for baseline computation
        fluo_path = os.path.join(self.experiment, "output", "axoid_internal",
                            "gui", "input_fluo.tif")
        if os.path.isfile(fluo_path):
            fluo_data = imread_to_float(fluo_path)
            tdtom, gcamp = get_fluorophores(fluo_data, identities)
        else:
            input_data = input_data.astype(np.float32) / input_data.max()
            tdtom, gcamp = get_fluorophores(input_data, identities)
        progress.setValue(77)
        dFF, dRR = compute_fluorescence(tdtom, gcamp, len_baseline)
        # Save results
        save_fluorescence(os.path.join(self.experiment, "output", "GC6_auto", "gui"),
                      tdtom, gcamp, dFF, dRR)
        progress.setValue(88)
        
        ## Re-load results in window
        self.combo_choice.activated[str].emit(self.combo_choice.currentText())
        self.applied_frames.setStringList(self.edited_frames.stringList())
        self.edited_frames.removeRows(0, self.edited_frames.rowCount())
        progress.setValue(100)
    
    def saveChanges(self):
        """Save changes applied to the entire experiment to the final/ folder."""
        if self.edited_frames.rowCount() > 0:
            msg_box = QMessageBox()
            msg_box.setText("Some frames were edited, but the changes were not applied.\n"
                            "Do you want to apply them first ?")
            msg_box.setStandardButtons(QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes)
            
            choice = msg_box.exec()
            if choice == QMessageBox.Cancel:
                return
            elif choice == QMessageBox.No:
                pass
            elif choice == QMessageBox.Yes:
                self.applyChanges()
        elif self.applied_frames.rowCount() == 0:
            return
        progress = self._get_progress_dialog("Saving changes...")
        
        # Copy current results into final/
        for i, folder in enumerate(["axoid_internal", "GC6_auto", "ROI_auto"]):
            final_path = os.path.join(self.experiment, "output", folder, "final")
            gui_path = os.path.join(self.experiment, "output", folder, "gui")
            if os.path.isdir(final_path):
                shutil.rmtree(final_path)
            shutil.copytree(gui_path, final_path)
            progress.setValue(15 * (i+1))
        
        # Reload folder data to display
        identities = io.imread(os.path.join(self.experiment, "output", 
                                            "axoid_internal", "final", "identities.tif"))
        self.identities["final"] = LabelStack(to_id_cmap(identities, cmap=ID_CMAP))
        self.rois["final"] = LabelStack(os.path.join(self.experiment, "output", 
                 "ROI_auto", "final", "RGB_seg.tif"))
        progress.setValue(80)
        self.combo_editChoice.activated[str].emit(self.combo_editChoice.currentText())
        self.combo_choice.activated[str].emit(self.combo_choice.currentText())
        
        # Reset frames
        self.applied_frames.removeRows(0, self.applied_frames.rowCount())
        progress.setValue(100)
    
    def _unsaved_changes(self):
        """Verify if there are unsaved changes, and ask the user to continue if so."""
        if self.edited_frames.rowCount() > 0 or self.applied_frames.rowCount() > 0:
            msg_box = QMessageBox()
            msg_box.setText("There are unsaved changes, are you sure you want to continue ?")
            msg_box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
            
            choice = msg_box.exec()
            if choice == QMessageBox.Ok:
                return False
            elif choice == QMessageBox.Cancel:
                return True
        else:
            return False
    
    def finish(self):
        """Finish the model correction and the AxoID GUI."""
        if not self._unsaved_changes():
            self.quitApp.emit()
    
    def gobackModel(self):
        """Return to the output selection page."""
        if not self._unsaved_changes():
            self.changedPage.emit(PAGE_MODEL)


class _EditStack(LabelStack):
    """Base class for editable stack through OpenCV, and display in PyQt5."""
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the label with the stack.
        
        Parameters
        ----------
        args : list of arguments
            List of arguments that will be passed to the LabelImage constructor.
        kwargs : dict of named arguments
            Dictionary of named arguments that will be passed to the LabelImage 
            constructor.
        """
        super().__init__(*args, **kwargs)
        self.index = 0
    
    def update_(self):
        """Redraw and scale the QPixmap and call QLabel update()."""
        self.pixmap_ = array2pixmap(self.stack[self.index])
        super().update_()
    
    def changeFrame(self, num):
        """
        Change the stack frame to display.
        
        Parameters
        ----------
        num : int
            Index of the frame in the stack to display.
        """
        self.index = num
        super().changeFrame(num)
    
    def event_coord(self, event):
        """Return the x and y coordinate of the event w.r.t. the image."""
        x, y = event.x(), event.y()
        # Substract the framebox
        x -= 1
        y -= 1
        # Transform from resized image to original image shape
        x = int(x * self.stack[0].shape[1] / self.frameGeometry().width())
        y = int(y * self.stack[0].shape[0] / self.frameGeometry().height())
        return x, y


class SegmentationStack(_EditStack):
    """Editable segmentation stack through OpenCV, and displayed in PyQt5."""


class IdentityStack(_EditStack):
    """Editable identity stack through OpenCV, and displayed in PyQt5."""
    editedFrame = pyqtSignal(int)
    
    def __init__(self, id_stack, *args, seg_stack=None, **kwargs):
        """
        Initialize the label with the stack.
        
        Parameters
        ----------
        id_stack : str or ndarray
            Path to or stack of identity images, that need to be converted 
            to a colormap before being displayed.
        args : list of arguments
            List of arguments that will be passed to the LabelImage constructor.
        seg_stack : str or  ndarray (optional)
            Path to or sttack of segmentation images used to potentially change
            IDs of discarded ROIs.
        kwargs : dict of named arguments
            Dictionary of named arguments that will be passed to the LabelImage 
            constructor.
        """
        self.OPACITY = 0.3
        self.COLOR = [127, 127, 127]
        
        if isinstance(id_stack, str):
            self.identities = io.imread(id_stack)
        elif isinstance(id_stack, np.ndarray):
            self.identities = id_stack
        if seg_stack is None:
            self.seg_stack = self.identities.astype(np.bool)
        if isinstance(seg_stack, str):
            self.seg_stack = io.imread(seg_stack)
        elif isinstance(seg_stack, np.ndarray):
            self.seg_stack = seg_stack
        
        self.mode = IDLE
        # List of list of changed frames
        self.changes = [[] for i in range(len(self.identities))]
        self.set_id = 0  # new identity after setID 
        
        self.n_axons = self.identities.max()
        stack = to_id_cmap(self.identities, cmap=ID_CMAP, vmax=self.n_axons)
        stack = overlay_mask_stack(stack, self.seg_stack, self.OPACITY, self.COLOR)
        
        super().__init__(stack, *args, **kwargs)
    
    def change_mode(self, mode):
        """
        Change the current drawing mode.
        
        Parameters
        ----------
        mode : int
            Drawing mode, see constants defined at the top of this file.
        """
        self.mode = mode
    
    def update_(self):
        """Update stack and displayed QPixmap."""
        if len(self.changes[self.index]) > 0:
            self.stack[self.index] = to_id_cmap(
                    self.changes[self.index][-1], cmap=ID_CMAP, vmax=self.n_axons)
            self.stack[self.index] = overlay_mask(
                    self.stack[self.index], self.seg_stack[self.index], self.OPACITY, self.COLOR)
        else:
            self.stack[self.index] = to_id_cmap(
                    self.identities[self.index], cmap=ID_CMAP, vmax=self.n_axons)
            self.stack[self.index] = overlay_mask(
                    self.stack[self.index], self.seg_stack[self.index], self.OPACITY, self.COLOR)
        super().update_()
    
    def reset(self):
        """Reset the current frame correction."""
        self.changes[self.index].clear()    
        self.update_()
    
    def undo_last(self):
        """Undo the last change."""
        if len(self.changes[self.index]) > 0:
            self.changes[self.index].pop()        
            self.update_()
    
    def set_set_id(self, str_id):
        """
        Set the identity that will be applied in case of setting.
        
        Parameters
        ----------
        str_id : str
            String of an ID number (e.g. str(1)), that can be casted as int.
        """
        self.set_id = int(str_id)
    
    def applyEditions(self):
        """Apply the currently edited frames to the identities stack."""
        for i, frames in enumerate(self.changes):
            if len(frames) > 0:
                self.identities[i] = frames[-1]
                frames.clear()
    
    def mousePressEvent(self, event):
        """Called when a mouse button is pressed over the image."""
        x, y = self.event_coord(event)
        if self.mode == IDLE or event.button() != Qt.LeftButton:
            return
        
        # Only if clicked on an axon
        if len(self.changes[self.index]) > 0:
            current_frame = self.changes[self.index][-1].copy()
        else:
            current_frame = self.identities[self.index].copy()
        if self.seg_stack[self.index][y, x]:
            # Look for single region to set the id
            tmp_labels = measure.label(self.seg_stack[self.index], connectivity=1)
            tmp_id = tmp_labels[y, x]
            
            if self.mode == SETID:
                current_frame[tmp_labels == tmp_id] = self.set_id + 1 # id actually starts at 1
            elif self.mode == DISCARDING:
                current_frame[tmp_labels == tmp_id] = 0
            self.changes[self.index].append(current_frame)
            self.editedFrame.emit(self.index)
        self.update_()


class ROIStack(_EditStack):
    """Editable ROI stack through OpenCV, and displayed in PyQt5."""