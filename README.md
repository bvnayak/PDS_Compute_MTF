# PDS_Compute_MTF - Slant Edge Method to Generate MTF curve.
Implementation of Slant Edge Method for MTF in Python from PDS Image.


Reference : http://www.mathworks.com/matlabcentral/fileexchange/28631-slant-edge-script/

# Features
* This code allows user to select ROI from the PDS Image.
* Detects Edge and Generates ESF, LSF and MTF Curves.

# Requirements
* cv2
* matplotlib
* scipy
* numpy

# Note
* Initial commit fails for some ROIs where the algorithm is unable to detect edge properly.
