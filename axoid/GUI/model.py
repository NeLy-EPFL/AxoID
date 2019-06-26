#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the model correction window widget.
This GUI page allows the user to modify the model of the tracker to improve the 
axon's identities (e.g.: by fusing, cutting, or discarding ROIs).
Created on Wed Jun 19 11:48:44 2019

@author: nicolas
"""

import os
import warnings
import pickle
import shutil

import numpy as np
from skimage import io
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QButtonGroup, QComboBox, QGroupBox, QHBoxLayout, 
                             QLabel, QMessageBox, QProgressDialog, QPushButton,
                             QVBoxLayout)
import cv2

from .constants import PAGE_SELECTION, PAGE_CORRECTION, CHOICE_PATHS, ID_CMAP, BIN_S, RATE_HZ
from .image import array2pixmap, LabelImage, LabelStack, VerticalScrollImage
from .multipage import AxoidPage
from axoid.tracking.cutting import fit_line, norm_to_ellipse, get_cut_pixels
from axoid.utils.image import to_npint, imread_to_float, to_id_cmap, overlay_mask
from axoid.utils.fluorescence import get_fluorophores, compute_fluorescence, save_fluorescence


## Constants
# Drawing modes
IDLE = 0
FUSING = 1
DISCARDING = 2
CUTDRAWING = 3


class ModelPage(AxoidPage):
    """
    Page of the model correction process.
    
    Here, the user can modify the model of the tracker to improve the 
    axon's identities (e.g.: by fusing, cutting, or discarding ROIs).
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the model page.
        
        Args and kwargs are passed to the AxoidPage constructor.
        """
        super().__init__(*args, **kwargs)
    
    def initUI(self):
        """Initialize the interface."""
        self.current_has_changed = False
        
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
        self.model_new = ModelImage(model_img)
        model_img = to_id_cmap(model_img, cmap=ID_CMAP)
        self.model = LabelImage(model_img)
        
        identities_path = os.path.join(self.experiment, "output", "axoid_internal",
                                       "final", "identities.tif")
        identities_img = to_id_cmap(io.imread(identities_path), cmap=ID_CMAP)
        self.identities = LabelStack(identities_img)
        self.identities_new = LabelStack(identities_img)
        for folder in ["final", "gui"]:            
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
        labels = ["Model", "Identities", "In folder", "Current"]
        positions = [(0,1), (0,2), (1,0), (2,0)]
        alignments = [Qt.AlignCenter] * 2 + [Qt.AlignVCenter | Qt.AlignLeft] * 2
        for label, position, alignment in zip(labels, positions, alignments):
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignCenter)
            self.left_display.addWidget(lbl, *position, alignment=alignment)
        
        # Model and identities images
        self.left_display.addWidget(self.model, 1, 1, alignment=Qt.AlignCenter)
        self.left_display.addWidget(self.model_new, 2, 1, alignment=Qt.AlignCenter)
        
        self.left_display.addWidget(self.identities, 1, 2, alignment=Qt.AlignCenter)
        self.left_display.addWidget(self.identities_new, 2, 2, alignment=Qt.AlignCenter)
        
        # Dropdown choice for last column
        self.combo_choice = QComboBox()
        self.combo_choice.setFocusPolicy(Qt.ClickFocus)
        self.combo_choice.addItems(["input", "segmentation", "rgb_init",
                                   "seg_init", "ROI_auto", "ΔR/R", "ΔF/F"])
        self.combo_choice.setCurrentText("rgb_init")
        self.changeChoice(self.combo_choice.currentText())
        self.combo_choice.activated[str].connect(self.changeChoice)
        self.left_display.addWidget(self.combo_choice, 0, 3, alignment=Qt.AlignCenter)
        # Last column images
        for folder, position in zip(["final", "gui"], [(1,3), (2,3)]):
            self.left_display.addLayout(self.choice_layouts[folder], *position)
        
        ## Right control with buttons and actions
        self.tool_buttongroup = QButtonGroup()
        # General toolbox
        general_vbox = QVBoxLayout()
        general_subbox = QHBoxLayout()
        general_groupbox = QGroupBox("General")
        reset_btn = QPushButton("Reset model")
        reset_btn.setToolTip("Reset the model to the one in the final/ folder")
        reset_btn.clicked.connect(self.resetModel)
        undo_btn = QPushButton("Undo")
        undo_btn.setToolTip("Undo the last change on the model")
        undo_btn.clicked.connect(self.undoAction)
        undo_btn.setShortcut("Ctrl+Z")
        for button in [reset_btn, undo_btn]:
            general_subbox.addWidget(button)
        general_groupbox.setLayout(general_subbox)
        general_vbox.addWidget(general_groupbox, alignment=Qt.AlignCenter)
        general_vbox.addStretch(1)
        # Identity modifications
        ids_vbox = QVBoxLayout()
        ids_subbox = QHBoxLayout()
        ids_groupbox = QGroupBox("Identities")
        fuse_btn = QPushButton("Fuse")
        fuse_btn.setToolTip("Select two identities and fuse them into a single one")
        fuse_btn.clicked[bool].connect(self.enableFuse)
        discard_btn = QPushButton("Discard")
        discard_btn.setToolTip("Discard (erase) the identity")
        discard_btn.clicked[bool].connect(self.enableDiscard)
        for button in [fuse_btn, discard_btn]:
            button.setCheckable(True)
            ids_subbox.addWidget(button)
            self.tool_buttongroup.addButton(button)
        ids_groupbox.setLayout(ids_subbox)
        ids_vbox.addWidget(ids_groupbox, alignment=Qt.AlignCenter)
        ids_vbox.addStretch(1)
        # ROI cutting
        cuts_vbox = QVBoxLayout()
        cuts_subbox = QHBoxLayout()
        cuts_groupbox = QGroupBox("Cutting")
        drawcut_btn = QPushButton("Draw cut")
        drawcut_btn.setToolTip("Draw a line representing a cut")
        drawcut_btn.clicked[bool].connect(self.enableDrawCut)
        drawcut_btn.setCheckable(True)
        self.tool_buttongroup.addButton(drawcut_btn)
        applycut_btn = QPushButton("Apply cut")
        applycut_btn.setToolTip("Apply the drawn cut to the model")
        applycut_btn.clicked.connect(self.applyCut)
        for button in [drawcut_btn, applycut_btn]:
            cuts_subbox.addWidget(button)
        cuts_groupbox.setLayout(cuts_subbox)
        cuts_vbox.addWidget(cuts_groupbox, alignment=Qt.AlignCenter)
        cuts_vbox.addStretch(1)
        
        
        # Apply/save/etc.
        fn_vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip("Apply changes made to model to current results")
        apply_btn.clicked.connect(self.applyChanges)
        save_btn = QPushButton("Save")
        save_btn.setToolTip("Save current results in the final/ folder")
        save_btn.clicked.connect(self.saveChanges)
        hbox.addWidget(apply_btn)
        hbox.addWidget(save_btn)
        finish_btn = QPushButton("Finish")
        finish_btn.setToolTip("Finish AxoID's user correction GUI with current results")
        finish_btn.clicked.connect(self.finish)
        correction_btn = QPushButton("Frame correction")
        correction_btn.setToolTip("Finish model correction and move on to frame correction")
        correction_btn.clicked.connect(self.gotoFrameCorrection)
        selection_btn = QPushButton("Back to selection")
        selection_btn.setToolTip("Return to the output selection page")
        selection_btn.clicked.connect(self.gobackSelection)
        
        fn_vbox.addLayout(hbox)
        fn_vbox.addWidget(finish_btn)
        fn_vbox.addWidget(correction_btn)
        fn_vbox.addStretch(1)
        fn_vbox.addWidget(selection_btn)
        
        self.right_title.setText("<b>Model correction</b>")
        self.right_control.addStretch(1)
        self.right_control.addLayout(general_vbox, stretch=1.5)
        self.right_control.addLayout(ids_vbox, stretch=1.5)
        self.right_control.addLayout(cuts_vbox, stretch=5)
        self.right_control.addLayout(fn_vbox, stretch=2)
    
    def changeFrame(self, num):
        """
        Change the stack frames to display.
        
        Parameters
        ----------
        num : int
            Index of the frame in the stacks to display.
        """
        super().changeFrame(num)
        self.identities.changeFrame(int(num))
        self.identities_new.changeFrame(int(num))
        for folder in ["final", "gui"]:
            if isinstance(self.choice_widgets[folder], LabelImage):
                self.choice_widgets[folder].changeFrame(int(num))
    
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
    
    def resetModel(self):
        """Reset the currently modified model to the model in final/ folder."""
        path = os.path.join(self.experiment, "output", "axoid_internal",
                            "final", "model.tif")
        model = io.imread(path)
        self.model_new.reset(model)
    
    def undoAction(self):
        """Undo the last change to the model."""
        self.model_new.undo_last()
    
    def enableFuse(self, checked):
        """Enable the fusing tool."""
        if checked:
            self.model_new.change_mode(FUSING)
    
    def enableDiscard(self, checked):
        """Enable the discarding tool."""
        if checked:
            self.model_new.change_mode(DISCARDING)
    
    def enableDrawCut(self, checked):
        """Enable the cut-drawing tool."""
        if checked:
            self.model_new.change_mode(CUTDRAWING)
    
    def applyCut(self):
        """Apply the currently drawn cut to the model."""
        self.model_new.applyCut()
    
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
        """Apply the modified model to the entire experiment."""
        if len(self.model_new.changes) == 0:
            return
        self.current_has_changed = True
        progress = self._get_progress_dialog("Applying changes...")
        
        ## Re-number ids from 1
        old_ids = np.unique(self.model_new.models[-1])
        old_ids = old_ids[old_ids != 0]
        new_ids = dict()
        for i, old_id in enumerate(old_ids):
            new_ids.update({old_id: i + 1})
        new_model = np.zeros_like(self.model_new.models[-1])
        for old_id, new_id in new_ids.items():
            new_model[self.model_new.models[-1] == old_id] = new_id
        
        # Save model
        path = os.path.join(self.experiment, "output", "axoid_internal",
                            "gui", "model.tif")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Warning)
            io.imsave(path, new_model)
        progress.setValue(20)
        
        ## Modify identities
        # Modify the frames one by one
        path = os.path.join(self.experiment, "output", "axoid_internal",
                            "gui", "identities.tif")
        identities = io.imread(path)
        for i in range(len(identities)):
            identities[i] = self.model_new.applyChanges(identities[i])
        progress.setValue(30)
        # Renumber ids
        new_identities = identities.copy()
        for old_id, new_id in new_ids.items():
            new_identities[identities == old_id] = new_id
        # Save results
        path = os.path.join(self.experiment, "output", "axoid_internal",
                            "gui", "identities.tif")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Warning)
            io.imsave(path, new_identities)
        progress.setValue(40)
        
        ## ROI_auto (raw data + contours)
        path = os.path.join(self.experiment, "output", "axoid_internal",
                            "gui", "input.tif")
        input_data = io.imread(path)
        ids = np.unique(new_identities)
        ids = ids[ids != 0]
        input_roi = to_npint(input_data)
        contour_list = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Warning)
            for i in range(len(input_roi)):
                contour_list.append([])
                for n, id in enumerate(ids):
                    roi_img = (new_identities[i] == id)
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
        progress.setValue(60)
        
        ## Fluorescence traces
        len_baseline = int(BIN_S * RATE_HZ + 0.5) # number of frames for baseline computation
        fluo_path = os.path.join(self.experiment, "output", "axoid_internal",
                            "gui", "input_fluo.tif")
        if os.path.isfile(fluo_path):
            fluo_data = imread_to_float(fluo_path)
            tdtom, gcamp = get_fluorophores(fluo_data, new_identities)
        else:
            io.imsave("/home/nicolas/Desktop/input_data.tif", input_data)
            input_data = input_data.astype(np.float32) / input_data.max()
            tdtom, gcamp = get_fluorophores(input_data, new_identities)
        progress.setValue(70)
        dFF, dRR = compute_fluorescence(tdtom, gcamp, len_baseline)
        # Save results
        save_fluorescence(os.path.join(self.experiment, "output", "GC6_auto", "gui"),
                      tdtom, gcamp, dFF, dRR)
        progress.setValue(80)
        
        ## Re-load results in window
        self.model_new.reset(new_model)
        self.identities_new.stack = to_id_cmap(new_identities, cmap=ID_CMAP, vmax=new_model.max())
        progress.setValue(90)
        self.identities_new.changeFrame(self.nframe_sld.value())
        self.combo_choice.activated[str].emit(self.combo_choice.currentText())
        progress.setValue(100)
    
    def saveChanges(self):
        """Save changes applied to the entire experiment to the final/ folder."""
        if not self.current_has_changed:
            return
        self.current_has_changed = False
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
        model_path = os.path.join(self.experiment, "output", "axoid_internal",
                                  "final", "model.tif")
        model_img = to_id_cmap(io.imread(model_path), cmap=ID_CMAP)
        self.model.pixmap_ = array2pixmap(model_img)
        self.model.update_()
        progress.setValue(60)
        
        identities_path = os.path.join(self.experiment, "output", "axoid_internal",
                                       "final", "identities.tif")
        identities_img = to_id_cmap(io.imread(identities_path), cmap=ID_CMAP)
        progress.setValue(75)
        self.identities.stack = identities_img
        self.identities.changeFrame(self.nframe_sld.value())
        progress.setValue(90)
        
        self.combo_choice.activated[str].emit(self.combo_choice.currentText())
        progress.setValue(100)
    
    def _unsaved_changes(self):
        """Verify if there are unsaved changes, and ask the user to continue if so."""
        if self.current_has_changed or len(self.model_new.models) > 1:
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
    
    def gotoFrameCorrection(self):
        """Finish the model correction, and go to the frame correction page."""
        if not self._unsaved_changes():
            self.changedPage.emit(PAGE_CORRECTION)
    
    def gobackSelection(self):
        """Return to the output selection page."""
        if not self._unsaved_changes():
            self.changedPage.emit(PAGE_SELECTION)


class ModelImage(LabelImage):
    """Editable model image through OpenCV, and displayed in PyQt5."""
    
    def __init__(self, image, *args, **kwargs):
        """
        Initialize the label with the image.
        
        Parameters
        ----------
        image : ndarray
            The image as a numpy array.
        args : list of arguments
            List of arguments that will be passed to the LabelImage constructor.
        kwargs : dict of named arguments
            Dictionary of named arguments that will be passed to the LabelImage 
            constructor.
        """
        # List of modified models, useful for the 'undo' action
        self.models = [image]
        self.mode = IDLE
        self.drawing = False
        self.cut_points = None
        self.cv2_overlay = np.zeros_like(self.models[-1])
        # Following are for transfering changes to the entire experiment
        #  - for fuse/discard: tuple of size 2 with old and new id
        #  - for cuts: tuple of size 3 with axon_id to be cut, tuple of cut's parameter,
        #    and the new id
        self.changes = []
        self._changes_num = []  # number of "changes" per action (useful for undo)
        
        rgb_image = to_id_cmap(image, cmap=ID_CMAP)
        
        super().__init__(rgb_image, *args, **kwargs)
    
    def update_(self, new_model=None):
        """
        Update models and displayed QPixmap.
        
        Parameters
        ----------
        new_model : ndarray (optional)
            Model image to append to the list of models.
        """
        if new_model is not None:
            self.models.append(new_model)
            self.cv2_overlay = np.zeros_like(self.cv2_overlay)
        rgb_image = to_id_cmap(self.models[-1], cmap=ID_CMAP)
        rgb_image = overlay_mask(rgb_image, self.cv2_overlay, 1, [255, 0, 0])
        self.pixmap_ = array2pixmap(rgb_image)
        super().update_()
    
    def reset(self, image):
        """
        Reset the model image correction.
        
        Parameters
        ----------
        image : ndarray
            Model image with which the model will be replaced.
        """
        self.models = [image]
        self.drawing = False
        self.cut_points = None
        self.cv2_overlay = np.zeros_like(self.models[-1])
        # Following are for transfering changes to entire experiment
        self.changes = []
        self._changes_num = []
        
        self.update_()
    
    def undo_last(self):
        """Undo the last change by removing the last model and its changes."""
        self.cut_points = None
        self.cv2_overlay = np.zeros_like(self.cv2_overlay)
        if len(self.models) > 1:
            for _ in range(self._changes_num.pop()):
                self.changes.pop()
            self.models.pop()
            self.update_()
    
    def applyCut(self):
        """Apply the current cut, if there is one."""
        if self.cut_points is not None:
            axon_id = np.max(self.models[-1][self.cv2_overlay.astype(np.bool)])
            roi_img = self.models[-1] == axon_id
            n, d = fit_line(self.cv2_overlay)
            n, d = norm_to_ellipse(n, d, roi_img)
            
            coords = get_cut_pixels(n, d, roi_img)
            new_model = self.models[-1].copy()
            new_id = self.models[-1].max() + 1
            new_model[coords[:,0], coords[:,1]] = new_id
            self.update_(new_model)
            
            self.changes.append((axon_id, (n,d), new_id))
            self._changes_num.append(1)
    
    def change_mode(self, mode):
        """
        Change the current drawing mode.
        
        Parameters
        ----------
        mode : int
            Drawing mode, see constants defined at the top of this file.
        """
        self.cv2_overlay = np.zeros_like(self.cv2_overlay)
        self.cut_points = None
        self.mode = mode
    
    def event_coord(self, event):
        """Return the x and y coordinate of the event w.r.t. the image."""
        x, y = event.x(), event.y()
        # Substract the framebox
        x -= 1
        y -= 1
        # Transform from resized image to original image shape
        x = int(x * self.models[-1].shape[1] / self.frameGeometry().width())
        y = int(y * self.models[-1].shape[0] / self.frameGeometry().height())
        return x, y
    
    def mousePressEvent(self, event):
        """Called when a mouse button is pressed over the image."""
        x, y = self.event_coord(event)
        if self.mode == IDLE or event.button() != Qt.LeftButton:
            return
        else:
            self.x = x
            self.y = y
        self.drawing = True
        self.update_()
    
    def mouseMoveEvent(self, event):
        """Called when the a mouse button is pressed and the mouse is moving."""
        x, y = self.event_coord(event)
        if self.mode == IDLE or not self.drawing:
            return
        elif self.mode in [FUSING, DISCARDING]:
            cv2.line(self.cv2_overlay, (self.x, self.y), (x, y), 1, 1)
            self.x, self.y = x, y
        elif self.mode == CUTDRAWING:
            self.cv2_overlay = np.zeros_like(self.cv2_overlay)
            cv2.line(self.cv2_overlay, (self.x, self.y), (x, y), 1, 1)
        self.update_()
    
    def mouseReleaseEvent(self, event):
        """Called when a mouse button is released."""
        x, y = self.event_coord(event)
        if self.mode == IDLE or event.button() != Qt.LeftButton:
            return
        elif self.mode in [FUSING, DISCARDING]:
            # Fuse or discard all label which have been selected
            new_model = self.models[-1].copy()
            ids = np.unique(self.models[-1][self.cv2_overlay.astype(np.bool)])
            ids = ids[ids != 0]
            if (self.mode == FUSING and len(ids) > 1) or \
               (self.mode == DISCARDING and len(ids) > 0):
                change_counter = 0
                for id in ids:
                    if self.mode == FUSING:
                        new_id = ids.min()
                    elif self.mode == DISCARDING:
                        new_id = 0
                    if id != new_id:
                        new_model[new_model == id] = new_id
                        self.changes.append((id, new_id))
                        change_counter += 1
                self._changes_num.append(change_counter)
                self.models.append(new_model)
            
            self.cv2_overlay = np.zeros_like(self.cv2_overlay)
        elif self.mode == CUTDRAWING:
            cv2.line(self.cv2_overlay, (self.x, self.y), (x, y), 1, 1)
            ids = np.unique(self.models[-1][self.cv2_overlay.astype(np.bool)])
            ids = ids[ids != 0]
            if len(ids) != 1:
                self.cv2_overlay = np.zeros_like(self.cv2_overlay)
            else:
                self.cut_points = ((self.x, self.y), (x, y))
        self.drawing = False
        self.update_()
    
    def applyChanges(self, id_frame):
        """
        Apply the saved changes to the given identity frame.
        
        Parameters
        ----------
        id_frame : ndarray
            Identity image on which to apply the changes performed on the model.
        
        Returns
        -------
        out_frame : ndarray
            New identity image with the applied changes.
        """
        out_frame = id_frame.copy()
        for change in self.changes:
            roi_img = out_frame == change[0]
            if np.sum(roi_img) == 0:
                continue
            if len(change) == 2:  # new id
                out_frame[roi_img] = change[1]
            elif len(change) == 3:  # cut
                n, d = change[1]
                coords = get_cut_pixels(n, d, roi_img)
                out_frame[coords[:,0], coords[:,1]] = change[2]
        return out_frame