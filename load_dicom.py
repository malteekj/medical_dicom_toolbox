import matplotlib.pyplot as plt
from matplotlib import cm
import pydicom
import os 
import numpy as np
import time
# plt.rcParams.update({'font.size': 22})

'''
Functions for loading, viewing and processing Dicom files
'''


def readDicom(dicomPath, shift_correct=[0,0,0], shift_units='pixels', rescale_units=True, window=None):
    '''
    Function for reading dicom files from a directory
    Input: 
        - dicomPath:        to the folder containing the dicom data
        - shift_correct:    correction in the x, y, and z plane
        - shift_units:      units of the shift correction
        - rescale_units:    Boolean for whether the image should be converted to HU
        - window:           Window to be applied to the image 
    Output: 
        - a 3D matrix contatining the image data
        - a list with the x, y and z coordinates of the image values
        - a list of the x, y and z coordinates of the patient position of each frame
    '''
    
    # Parameters:
    print('rescale conversion: ' + str(rescale_units))
    print('Window: ' + str(window))
    
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(dicomPath):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))
    
    # Get ref file
    RefDs = pydicom.read_file(lstFilesDCM[0])
    
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
    
    print('pixel spacing', ConstPixelSpacing)
    
    # Apply for a correction for the coordinate system
    if shift_units == 'pixels':
        shift_correct = np.array(shift_correct) * np.array(ConstPixelSpacing)
    elif shift_units == 'mm':
        pass 
    
    # Create coordinates before correction
    x = np.arange(0.0, float((ConstPixelDims[0]))*ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, float((ConstPixelDims[1]))*ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0, float((ConstPixelDims[2]))*ConstPixelSpacing[2], ConstPixelSpacing[2])
    
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=np.int32)
    
    patient_position = np.empty((len(lstFilesDCM),3))
    
    print('Loading image data..')
    # loop through all the DICOM files
    for i, filenameDCM in enumerate(lstFilesDCM):
        if i%100 == 0:
            print(i,'/',len(lstFilesDCM))
        # read the dicom-file file
        ds = pydicom.read_file(filenameDCM)
        
        # Instance number is used place slides in the right order
        if rescale_units:
            image_temp = ds.pixel_array*ds.RescaleSlope + ds.RescaleIntercept
            # Aplly window if wanted
            if window is not None:
                image_temp[image_temp < window[0]] = window[0]
                image_temp[image_temp > window[1]] = window[1]

            ArrayDicom[:, :, int(ds.InstanceNumber)-1] = image_temp
        else:
            ArrayDicom[:, :, int(ds.InstanceNumber)-1] = ds.pixel_array 
        
        # Load the orientation of the patient
        patient_orientation = np.array(ds.ImageOrientationPatient)
        # # Correct for the Image orientation Xx*i + Yx*j etc.
        # x_rot = x[int(ds.InstanceNumber)-1]*patient_orientation[0]+y[int(ds.InstanceNumber)-1]*patient_orientation[3] 
        # y_rot = x[int(ds.InstanceNumber)-1]*patient_orientation[1]+y[int(ds.InstanceNumber)-1]*patient_orientation[4] 
        
        # x[int(ds.InstanceNumber)-1] = x_rot
        # y[int(ds.InstanceNumber)-1] = y_rot
        
        # Get the reference pixel
        patient_position[int(ds.InstanceNumber)-1,:] = np.array(ds.ImagePositionPatient)
        
        # # Correct the coordinates for the Image Position Patient
        # pos_temp = np.array(ds.ImagePositionPatient)
        # x[int(ds.InstanceNumber)-1] += pos_temp[0] + shift_correct[0]
        # y[int(ds.InstanceNumber)-1] += pos_temp[1] + shift_correct[1]
        # z[int(ds.InstanceNumber)-1] = pos_temp[2] + shift_correct[2] # the z axis is just the position of the slice
    
    # Flip the x dimension
    
    print('The coordinate system was shift corrected: [%.2f,%.2f,%.2f] mm' % (shift_correct[0], shift_correct[1], shift_correct[2]))
    print('done.\n')
    ArrayDicom = np.flip(ArrayDicom, axis=0)
    return ArrayDicom, patient_orientation, patient_position


'''
Function for extracting ROIs 
'''
def extractROIs(file_path,patient_orientation=None, coordsCT=None, verbose=False, return_spacing=False, reverse_axis=[False]):
    '''
    Extract rois from a dicom file and fills the outline with points. If no coordsCT is given,
    It's assumed that the ROI is from Inveon and hence the positions are stored in private tags
    Input:
        file_path:                  Path to the .dcm file with the ROIs                
        patient_orientation:        Orientation of the patient coordinate system, list with 6 entries: x,y and z for x axis and y axis
        coordsCT:                   Coordinate system of the patient
        verbose:                    Whether to print out all the rois at the ent
    Output:
        rois:                       dict with the name, coords_mm and coords_pix (filled)
    '''
    
    print('loading rois..')
    # Load data
    roi_dcm = pydicom.read_file(file_path)
    
    # Construct the coordinate system
    if coordsCT is None:
        # private tags from Inveon for coordinate reference
        ConstPixelDims = roi_dcm['0x007d1001'].value
        patient_position = roi_dcm['0x007d1003'].value
        ConstPixelSpacing = roi_dcm['0x007d1002'].value 
       
        x = np.arange(0.0, float((ConstPixelDims[0]))*ConstPixelSpacing[0], ConstPixelSpacing[0])-patient_position[0]
        y = np.arange(0.0, float((ConstPixelDims[1]))*ConstPixelSpacing[1], ConstPixelSpacing[1])-patient_position[1]
        z = np.arange(0.0, float((ConstPixelDims[2]))*ConstPixelSpacing[2], ConstPixelSpacing[2])-patient_position[2]
        
        if patient_orientation is not None:
            # projection onto the the x and y basis vector of patient_orientation
            x_rot = x*patient_orientation[0] + y*patient_orientation[3]
            y_rot = x*patient_orientation[1] + y*patient_orientation[4]
            
            x = x_rot 
            y = y_rot
        
        # Some coordinate systems have different positive directions on axis
        if 'x' in reverse_axis:
            x = np.flip(x)
            print('Reversed x-axis')
        elif 'y' in reverse_axis:
            y = np.flip(y)
            print('Reversed y-axis')
        elif 'z' in reverse_axis:
            z = np.flip(z)
            print('Reversed z-axis')

            
        coordsCT = [x,y,z]        
        
    # Dict containing each data and info abput each roi
    rois = {}
    # iterate over each roi
    for roi_num in range(len(roi_dcm.StructureSetROISequence)):
        if (roi_num+1) % 10 == 0 or roi_num==0:
            print(roi_num+1,'/',len(roi_dcm.StructureSetROISequence))
            
        # iterate over each slice in the ROI
        coords_mm = []
        roi_pixel_placement = []
        
        # extract all the pixel values in mm
        # pixels are contained as [x1,y1,z1,x2,y2,z2,...]
        for roi_slice in roi_dcm.ROIContourSequence[roi_num].ContourSequence:
            coords_mm.append(np.array(roi_slice.ContourData).reshape(-1,3))
            
        # calculate the coordinates in pixels 
        for coord_item in coords_mm:
            roi_pixel_placement_temp = np.empty(np.shape(coord_item))
            
            for i, coord in enumerate(coord_item):
                # find the closest x, y and z coordinate
                x_min = np.argsort(np.abs(coordsCT[0]-coord[0]))[0]
                y_min = np.argsort(np.abs(coordsCT[1]-coord[1]))[0]
                z_min = np.argsort(np.abs(coordsCT[2]-coord[2]))[0]
                roi_pixel_placement_temp[i,:] = np.array([x_min, y_min, z_min])
            
            roi_pixel_placement.append(roi_pixel_placement_temp)
            
        rois[roi_num] = {'name': roi_dcm.StructureSetROISequence[roi_num].ROIName,
                       'coords_mm': coords_mm,
                       'coords_pix': roi_pixel_placement}
    if verbose:
        print('\nSeries description: ' + roi_dcm.SeriesDescription)
        print('Number of ROIs:', len(roi_dcm.StructureSetROISequence))
        for i in range(len(rois)):
            print('{:<4} name: {}, number of slices: {}'.format('('+str(i)+')', rois[i]['name'], len(rois[i]['coords_pix']) ))
    if return_spacing:
        return rois, ConstPixelSpacing
    else:
        return rois 

class multi_slice_viewer:
    def __init__(self,volume, second_volume=None, second_cmap='hot', alpha=0.5, rois=None,
                 start_slice=450,shift_correct=[0,0], window=[None, None], second_window=[None, None],
                 cmap='gray', orientation='axial'):
        self.rois = rois
        self.remove_keymap_conflicts({'up', 'down','j','l'})
        self.fig, self.ax = plt.subplots(figsize=(10,7.5))
        self.second_volume = second_volume
        self.orientation = orientation
        
        # handle the orientation of the image
        if orientation == 'axial':
            self.ax.volume = volume
            if second_volume is not None:
                self.ax.sec_volume = second_volume
        elif orientation == 'sagittal':
            self.ax.volume = np.transpose(volume, (0,2,1))
            if second_volume is not None:
                self.ax.sec_volume = np.transpose(second_volume, (0,2,1))
        elif orientation == 'coronal':
            # self.ax.volume = np.flip(np.transpose(volume, (2,1,0)),axis=0)
            self.ax.volume = np.transpose(volume, (2,1,0))
            if second_volume is not None:
                # self.ax.sec_volume = np.flip(np.transpose(second_volume, (2,1,0)),axis=0)
                self.ax.sec_volume = np.transpose(second_volume, (2,1,0))
        else:
            raise KeyError('Orientation has to be either: \'axial\', \'sagittal\' or \'coronal\'') 
            
        self.ax.rois = rois
        self.ax.index = start_slice
        
        # for testing correction
        self.x_shift = shift_correct[0]
        self.y_shift = shift_correct[1]
        
        self.ax.imshow(self.ax.volume[:,:,self.ax.index], cmap=cmap, origin='lower', vmin=window[0], vmax=window[1])
        if self.second_volume is not None:
            self.alpha = alpha
            self.ax.imshow(self.ax.sec_volume[:,:,self.ax.index], cmap=second_cmap, origin='lower', alpha=self.alpha, vmin=second_window[0], vmax=second_window[1])
            
        self.fig.canvas.mpl_connect('key_release_event', self.process_key)
        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.fig.canvas.mpl_connect('scroll_event', self.process_key)
        
        self.ax.set_title(self.ax.index)
        
        self.ax.drawed_roi = []
        if self.ax.rois is not None: 
            self.draw_rois()
        
        self.slice_jump = 1
        
        plt.show()
            
            
    def process_key(self, event):

        # Release ctrl key
        if event.name == 'key_release_event' and event.key == 'control':
            self.slice_jump = 1
        
        if event.name == 'key_press_event':
            # For bigger jumps
            if event.key == 'control':
                self.slice_jump = 10
            elif event.key == 'up':
                self.previous_slice(self.slice_jump)
            elif event.key == 'down':
                self.next_slice(self.slice_jump)
            elif event.key == 'j':
                self.decrease_alpha()
            elif event.key == 'l':
                self.increase_alpha()
        elif event.name == 'scroll_event':
            if event.button == 'up':
                self.previous_slice(self.slice_jump)
            elif event.button == 'down':
                self.next_slice(self.slice_jump)
                
        self.ax.set_title(self.ax.index)
        self.fig.canvas.draw()
    
    def decrease_alpha(self):
        self.alpha -= 0.05
        if self.alpha < 0:
            self.alpha = 0
        self.ax.images[1].set_alpha(self.alpha)
    
    def increase_alpha(self):
        self.alpha += 0.05
        if self.alpha > 1:
            self.alpha = 1
        self.ax.images[1].set_alpha(self.alpha)
     
    def previous_slice(self, jump):
        # volume = ax.volume
        self.ax.index = (self.ax.index - jump) % self.ax.volume.shape[2]  # wrap around using %
        self.ax.images[0].set_array(self.ax.volume[:,:,self.ax.index])
        # Drawing the second volume 
        if self.second_volume is not None:
            self.ax.images[1].set_array(self.ax.sec_volume[:,:,self.ax.index])
            
        # find roi if any
        for drawing in self.ax.drawed_roi:
            drawing[0].remove()
        
        self.ax.drawed_roi = []
        if self.rois is not None:
            self.draw_rois()
            
    def next_slice(self, jump):
        self.ax.index = (self.ax.index + jump) % self.ax.volume.shape[2]  # wrap around using %
        self.ax.images[0].set_array(self.ax.volume[:,:,self.ax.index])
        # Drawing the second volume 
        if self.second_volume is not None:
            self.ax.images[1].set_array(self.ax.sec_volume[:,:,self.ax.index])
       
        # find roi if any 
        for drawing in self.ax.drawed_roi:
            drawing[0].remove()
        
        self.ax.drawed_roi = []
        if self.rois is not None:
            self.draw_rois()
            
    def draw_rois(self):
        # Function for drawing the rois
        for roi in self.ax.rois: # a list rois for each structure
             for roi_slice in roi: # each slice in that roi
                 if self.orientation == 'axial':
                     if int(roi_slice[0,2]) == self.ax.index: # draw if z coordinate is the current slice
                         self.ax.drawed_roi.append(self.ax.plot(roi_slice[:,0]+self.x_shift, roi_slice[:,1]+self.y_shift,'r*'))
                 if self.orientation == 'coronal':
                     idx_temp = np.where(roi_slice[:,1] == self.ax.index)[0]
                     self.ax.drawed_roi.append(self.ax.plot(roi_slice[idx_temp,0]+self.x_shift, roi_slice[idx_temp,2]+self.y_shift,'r*'))
              
    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)


'''
Wrapper for filling rois
'''
def fillRoi(points, make_mask=False, size=None, correct_border=False):
    '''
    Wrapper for calling function to space fill the ROIs  
    Input:
        points:             Nx2 numpy array of x,y coordinates
        make_mask:          Make a binary mask of the coordinates with specified size
        size:               size of binary mask (tuple), if make_mask is True
        correct_border:     whether weight the border area by center places coordinates
    Output:
        filled_poits:   filled points either as x,y coordinates or as binary mask
    '''    
    interpolated_points = intpolPoints(points)
    filled_points = fillPoints(interpolated_points).astype(np.int16)
    
    # Create mask if specified 
    if make_mask:
        if size is None:
            raise ValueError('\'Size\' has to be specified when \'make_mask\' is True')
        mask = np.zeros(size, dtype=np.float)
        mask[filled_points[:,1], filled_points[:,0]] = 1
        
        if correct_border:
            points_075, points_050, points_025 = findBorder(mask)
            mask[points_075[:,0],points_075[:,1]] = 0.75
            mask[points_050[:,0],points_050[:,1]] = 0.5
            mask[points_025[:,0],points_025[:,1]] = 0.25
        return mask
    else:
        return filled_points.astype(np.int16)
    
def intpolPoints(points):
    '''
    This function takes the ROI points as input and interpolates the missing points between them where needed
    
    Input:
        points:                 input ROI points, numpy array (Nx2)
    Output:
        interpolated_points:    interpolated poits numpy array (N+Mx2)
    '''
    
    x_prev = None
    y_prev = None
    
    interpolated_points = np.empty((0,2))
    
    for i, (x,y) in enumerate(np.concatenate((points,points[-1,:].reshape(1,2)),axis=0)):
            
        if x_prev is None:
            pass # Just jump to the bottom
        else:
            if x_prev == x:
                y_diff = y-y_prev
                # add missing points
                if y_diff > 1: 
                    for k in np.arange(1,y_diff):
                        interpolated_points = np.concatenate((interpolated_points, np.array([[x_prev,y_prev+k]])), axis=0).astype(int)
                elif y_diff < -1:
                    for k in np.arange(1,abs(y_diff)):
                        interpolated_points = np.concatenate((interpolated_points,np.array([[x_prev,y_prev-k]])), axis=0).astype(int)
            
            if y_prev == y:
                x_diff = x-x_prev
                # add missing points
                if x_diff > 1: 
                    for k in np.arange(1,x_diff):
                        interpolated_points = np.concatenate((interpolated_points, np.array([[x_prev+k,y_prev]])), axis=0).astype(int)
                elif x_diff < -1:
                    for k in np.arange(1,abs(x_diff)):
                        interpolated_points = np.concatenate((interpolated_points,np.array([[x_prev-k,y_prev]])), axis=0).astype(int)
            
        # add the points that already was in the set        
        interpolated_points = np.concatenate((interpolated_points,np.array([[x,y]])), axis=0)
        
        x_prev = x
        y_prev = y
    
    # remove dublicates from the very last points 
    interpolated_points = interpolated_points[:-2,:]    
    
    return interpolated_points.astype(np.int16)

def fillPoints(interpolated_points):
    '''
    This function takes the ROI points as input and returns all points within the roi
    
    Input:
        points:                 input ROI points, numpy array (Nx2)
    Output:
        interpolated_points:     interpolated poits numpy array (N+Mx2)
    '''
    # Find the boundaries of the grid to search
    x_min, x_max = np.min(interpolated_points[:,0]), np.max(interpolated_points[:,0])
    y_min, y_max = np.min(interpolated_points[:,1]), np.max(interpolated_points[:,1])
    
    fill_points = np.empty((0,2))
    
    # search within the boundaries (x-1)
    for point_x in np.arange(x_min,x_max+1):
        for point_y in np.arange(y_min,y_max+1):
            
            # Check if point is on the edge 
            if (interpolated_points == np.array([point_x,point_y])).all(axis=1).any():
                fill_points = np.concatenate((fill_points, np.array([point_x, point_y]).reshape(1,2).astype(int)),axis=0)
            # Check if inside
            else:
                scan_y = interpolated_points[interpolated_points[:,1] == point_y,0]
                if any(scan_y>point_x) and any(scan_y<point_x):
                    fill_points = np.concatenate((fill_points, np.array([point_x, point_y]).reshape(1,2).astype(int)),axis=0)
    return fill_points

def findBorder(filled_points_mask):
    '''
    input is expected to be a one slice mask for the roi
    output: The x, y coordinates for where to place 0.25, 0.5 and 0.75
    '''
    
    x_min, x_max = np.min(np.where(np.sum(filled_points_mask,axis=1)>0)), np.max(np.where(np.sum(filled_points_mask,axis=1)>0))
    y_min, y_max = np.min(np.where(np.sum(filled_points_mask,axis=0)>0)), np.max(np.where(np.sum(filled_points_mask,axis=0)>0))
    
    # scannning alon the x axis starting from outermost value of the roi
    points_025 = []
    points_050 = []
    points_075 = []
    
    # loop over grid 
    for y in np.arange(y_min,y_max+1):
        for x in np.arange(x_min,x_max+1):
            if filled_points_mask[x,y]==1:
                sum_3x3 = np.sum(filled_points_mask[x-1:x+2,y-1:y+2]) # check the neighborhood 
                if sum_3x3 == 8:
                    points_075.append([x,y])
                elif sum_3x3 == 7:
                    points_050.append([x,y])
                elif sum_3x3 == 6:
                    if any(np.sum(filled_points_mask[x-1:x+2,y-1:y+2],axis=0)==0) or any(np.sum(filled_points_mask[x-1:x+2,y-1:y+2],axis=1)==0): 
                        points_050.append([x,y])
                    else: 
                        points_025.append([x,y])
                elif sum_3x3 < 6:
                    points_025.append([x,y])             

    points_025 = np.array(points_025)
    points_050 = np.array(points_050)
    points_075 = np.array(points_075)
    
    return points_075, points_050,  points_025


