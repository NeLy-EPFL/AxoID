## How to use the GUI
Here is the manual of the different GUI pages. The GUI load data when changing pages, therefore it might be slightly slow.

**Note:** if you run the GUI through `ssh`, you might need the option `-X` or `-Y` to forward the window to your screen. Additionally, note that it might be slower than running the GUI from your own machine.

### Output selection
The first page of the GUI. Here, the user can explore the outputs of the main *AxoID* pipeline on the raw data (*raw*), the cross-correlation registered data (*ccreg*), and the optic flow warped data (*warped*). The goal here is to choose one of these outputs as the "final" output. It can be further corrected on the following pages.

![](../images/gui_selection.png "Output selection page of the GUI")

The columns display the 3 different output (raw, ccreg, and warped). Each row corresponds to a different type of data:
  * the first one is the tracker model, i.e., how the tracker thinks the axons are organised. **1** Indicates the color mapping between axon identity and color (valid for model and identities)
  * the second row displays the contours of the ROIs over the input data
  * the last row can display different data, based on the dropdown choice **2** (see [Running AxoID](./Running.md) for description of the outputs)  
Under these, a slider **3** can be used to change the index of the frame being currently displayed.

Once the user has chosen an output to continue with, the menu **4** can be used:
  * the *Choose output* dropown is used to select one of raw, ccreg, or warped (note that if you already used the GUI on the experiment before, you will be able to continue working with the previous data by choosing *final* in this menu)
  * once a correct output is chosen, the *Select output* button should be enabled. Clicking it will save the chosen output to the `final/` folders (or do nothing if *final* was chosen), and the GUI will continue to the model correction page
  * **(not implemented, see below)** if no data is satisfactory, the user can rather choose to manually annotate a few frames and use these for fine tuning the network. The rest of the *AxoID* pipeline will be applied

### Model correction
This GUI page lets the user modify the tracker model, and then transfer the changes to the entire experiment. Available tools consist in fusing existing ROIs which would correspond to the same axon, discarding useless ROIs or misdetection (as trachea), and applying "cuts" to separate touching axons (see ROI cutting in [Pipeline](./Pipeline.md)).

![](../images/gui_model.png "Model correction page of the GUI")

The left display is now separated in two rows:
  * the top row corresponds to the data in the `final/` folders
  * the bottom row is the data being currently modified

The three columns display:
  * the tracker model
  * the identities of ROIs on the currently displayed frame (see slider **3**), the colors of the ROIs indicate their identity w.r.t. their model
  * different kind of data, where the **1** dropdown menu can be used to select which one to show

This page allows the user to modify the tracker model displayed in **2**, using the tools of the menu **4**. The different tools are:
  * General
    * *Reset model*: reset the model to the one in the `final/` folder (i.e., the one displayed in the top row)
    * *Undo*: undo the last change made to the model
  * Identities
    * *Fuse*: enable the fusing tool. In this mode, drawing over the model displays a red line which automatically select the ROIs that it has been drawn upon. When the left mouse button is released, all selected ROIs will be fused, effectively being set to the same ID. E.g.: if you want to fuse ROI 0 and ROI 1, select this tool, then draw a line passing over the two ROIs and then release the button
    * *Discard*: enable the discard tool. In this mode, drawing over an ROI will also select it, and when the left mouse button is released, all selected ROIs will be discarded (effectively setting them to background)
  * Cutting
    * *Draw cut*: enable the drawing of a cut. In this mode, the user can draw a "cut" on the model, a simple segment representing the cutting border of the ROI. It requires cutting one and only one ROI (if the cut is valid, the shown red segment will stay on the image as long as the *Draw cut* tool is selected).
    * *Apply cut*: apply the drawn cut to the model image, effectively making two ROIs out of the cutted one. This only works if a valid cut is drawn on the current model image

Note that all these changes are currently only applied to the model only. To apply them to the whole experiment, and then save the results, use the menu **5**:
  * *Apply*: apply the model changes to the entire experiment, also recomputing the fluorescence traces
  * *Save*: when the user is satisfied with the current results, this button save them to the `final/` folder, effectively over-writting the previous one (the top row will be updated with the newly saved data)
  * *Frame correction*: finish the model correction, and move on to the frame correction page. To be used when the user is satisfied with the saved results, but desires to edit some frames
  * *Back to selection*: go back to the output selection page
  * *Finish*: finish the model correction, and also terminate the GUI. To be used when the user is satisfied with the saved results, and does not want to edit single frames

### Frame correction
This page lets the user modify single frames of the results. It is possible to change the identity of the detected ROIs, and redraw/replace the ROIs over the frames. As before, it works by editing current results, then applying them, finally saving them on disk when satisfied.

![](../images/gui_correction.png "Frame correction page of the GUI in identity edition mode")

The left display is similar to the model correction page, with few adjustment:
  * there is only one common model between the data in folder and the data being currently modified (see **1** for the common color mapping between ROI identity and color).
  * the second column is now where the editions take place (see **2**), and the different options can be chosen using the **4** dropdown menu. Note that changing this choice will also adapt the toolboxes **6**, see below:
    * identities: modify the identity of detected ROI
    * ROI: redraw the ROI over the frame
  * last column is still for displaying different kind of data, with the **5** dropdown menu

The slider **3** should be used to change the frame to be currently displayed and edited. The toolbox in **6** give access to the different tools to edit the frame, depending on the choice of **4**:
  * identities: the identity frame is displayed, with the segmentations overlayed ontop as light gray. Detected ROIs without identity can still be used with the following tools:
    * *Reset*: reset the identity frame to the one in the `final/` folder
    * *Undo*: undo the last change performed on the frame
    * *Set ID*: enable the tool to assign identity. Use the dropdown menu right of it to select the identity that you want to assign, and then click on the ROI that you want to set its ID. It can be used to exchange the ID of two ROIs by sequentially setting their identities to the other's
    * *Discard*: enable the discarding tool. With this tool enabled, click on an ROI to discard it (setting its identity to 0 -background-)
  * ROI: TODO + image

![](../images/gui_correction_ROI.png "Frame correction page of the GUI in ROI edition mode")

The display in **7** shows which frame has been edited (left), and which frame has been edited AND this edition has been applied to recompute the results (e.g. fluorescence) (right). Note that current version of the GUI (as of 1st July 2019) keeps edited frame in this display, even if the frame has been *reset* to the one in folder.

Similarly to last page, the tools in **8** allows for:
  * *Apply*: apply the editions made to the frames ("Edited frames" in **7**) to the results, and recompute fluorescence accordingly
  * *Save*: save the applied results ("Applied frames" in **7**) to the folder, effectively over-writting the `final/` folder (top row in the display will be updated accordingly)
  * *Finish*: finish the frame correction, and terminate the GUI. To use when the user is satisfied with the final results in folder
  * *Back to model*: go back to the model correction page
  * *Manual annotation*: **(not implemented, see below)** if results are not satisfying, go to the manual annotation page

### Manual annotation (not implemented)

![](../images/gui_annotation.png "Manual annotation mini-GUI")
