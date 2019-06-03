import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from scipy import ndimage
import os
import cv2
from skimage import transform as tf
from skimage import measure
from skimage import filters
from scipy import stats
from collections import OrderedDict
import shutil
#from pathlib import Path
#from wfdb import wrsamp
#from cardio import EcgDataset
#from cardio.pipelines import hmm_predict_pipeline
#from cardio import ecg_batch_tools as bt
#import warnings
#warnings.filterwarnings('ignore')

def find_peaks(a, spacing, border, numpeaks, descending=True):
    """
    For a 1-D array 'a', find_peaks looks for 'numpeaks' # of maximum values
    that are separated from eachother by a distance 'spacing' and are a distance
    'border' away from the edges. Returns 'allidx', the indices of the peak values.

    INPUTS:

    a - 1-D array (pixel sums in the horizontal direction)

    spacing - how far away peaks must be from each other. It's likely to have
        high peaks clustered that refer to the same region. This makes sure that
        the returned peaks are spaced out by at least this distance.

    border - distance from the edges of the array to exclude. For this application
        the peak values should never occur near the edges, so this removes any
        error due to pronounced artifacts occuring towards the edges of the image.

    numpeaks - the number of peaks to return. For the current data crop, it should
        be set to 4. This is equivalent to saying how many rows of data are in the
        image.

    descending - search for lowest values or highest values. True means find highest
        values. False means find lowest values.

    OUTPUTS:

    allidx - an array containing the indices of the peak values in array 'a'

    This is used to find center lines of the data. The center line is used to
    find the crop borders of each lead's waveform.
    """
    allidx = -1*np.ones(numpeaks, dtype=np.int)

    if descending:
        a[:border] = 0
        a[-border:] = 0
        argsort = np.argsort(a)[::-1]
    else:
        a[:border] = np.max(a)
        a[-border:] = np.max(a)
        argsort = np.argsort(a)
    peaknum = 0
    idx = 0
    while idx < len(a) and peaknum < numpeaks:
        delta = np.abs(argsort[idx] - argsort[:idx])

        if np.all(delta > spacing):
            allidx[peaknum] = argsort[idx]
            peaknum += 1
            idx += 1
        else:
            idx += 1

    return allidx

def remove_spots(mask, thresh):
    """
    remove_spots uses ndimage.label to remove any blobs from the binary mask
    'mask' that have an area smaller than 'thresh'.

    INPUTS:

    mask - a 2-D binary mask to be cleaned.

    thresh - the minimum size of a blob to be included. Any contiguous regions
        in the binary mask smaller than 'thresh' are removed

    OUTPUTS:

    outmask - the cleaned binary mask
    """
    outmask = np.zeros(mask.shape)
    larr, numf = ndimage.label(mask)

    sums = ndimage.sum(mask,larr,range(1,numf+1))

    goodnums = np.arange(1,numf+1)[sums>thresh]

    for ii in goodnums:
        outmask += larr == ii

    return outmask


def extract_waveform(mask, remove_spikes=False):
    """
    For a given cleaned binary mask, 'mask', representing an image of a waveform,
    extract_waveform extracts the closest underlying 1-D waveform and returns
    this as a numpy array.

    INPUTS:

    mask - a cleaned binary mask of the waveform.

    remove_spikes - gets rid of any spikes caused by noisy segmentaion.

    OUTPUTS:

    ts - a numpy array of the 2-D waveform transformed to 1-D. (By 2-D, I mean
        it's in image coordinates and 1-D I mean it returns the Y value for
        each pixel in the horizontal dimension of the image.)

    extract_waveform takes each vertical slice of the 2-D input and grabs the
    indices of the nonzero pixels in that slice. Due to the noise in the image,
    it returns the average Y value for the longest continuous non-zero region
    in the slice.

    NOTE: This has a tendency to dull peaks, as the line
    thickness of the printout can bleed peaks into many vertical pixels in the
    same slice. As an update, we could include information about the location
    and derivative of the function to better handle peaks.
    """
    ts = np.zeros(mask.shape[1])
    for ii in range(mask.shape[1]):
        tslice = mask[:,ii]

        if tslice.sum() > 0:

            tdiff = np.append(0,np.diff(tslice))

            ids, = np.where(tdiff == 1)
            ide, = np.where(tdiff == -1)

            if ids.shape[0] < ide.shape[0]:
                ide = ide[:ids.shape[0]]
            elif ide.shape[0] < ids.shape[0]:
                ids = ids[:ide.shape[0]]

            idmax = np.argmax(ide - ids)

            ts[ii] = abs(ide[idmax] - ids[idmax])/2. + ids[idmax]
    ts[ts==0] = np.nan

    if remove_spikes:
        wavediff = np.diff(np.append(0,ts))
        wavediff[np.isnan(wavediff)] = 0

        ts[np.abs(wavediff)>25] = np.nan
        # wavediff = np.append(wavediff,np.array([0,0,0]))
        #
        # # this could be vectorized by shifting the diff array
        # for ii in range(wavediff.shape[0]):
        #     if np.abs(wavediff[ii]) > 20 and np.abs(wavediff[ii+1]) > 20 and np.sign(wavediff[ii]/wavediff[ii+1]) == -1:
        #         ts[ii:ii+2] = np.nan
        #     elif np.abs(wavediff[ii]) > 20 and np.abs(wavediff[ii+2]) > 20 and np.sign(wavediff[ii]/wavediff[ii+2]) == -1:
        #         ts[ii:ii+3] = np.nan

    return ts

def get_contours(mask, arcsize=0.005):
    """
    For a provided mask and arcsize, this will find the best-fit polygon. For images
    with the patient-sensitive data removed via cutting, this polygon will not be
    a quadrilateral. This returns an array of (x,y) coordinates for each vertex.

    INPUTS:

    mask - a binary mask representing the extent of the EKG within the full image

    arcsize - the ratio of the shape's perimeter that is considered a separate edge.
                A larger number means a side must be larger to be considered separate.
                This is useful if there are many small concave gaps in a side that
                should be ignored to get the larger shape.

    OUTPUTS:

    approx - a numpy array of x,y coordinates for each vertex of the determined shape.
    """
    contours,hierarchy = cv2.findContours((mask*255).astype(np.uint8), 1, 2)


    areas=[]
    for c in contours:
    	areas.append(cv2.contourArea(c))

    imax = np.argmax(np.asarray(areas))

    cnt = contours[imax]

    epsilon = arcsize*cv2.arcLength(cnt,True)

    approx = cv2.approxPolyDP(cnt,epsilon,True)

    return approx


def transform_pic(pic):
    """
    transform_pic extracts the EKG from the image and stretches/skews/rotates it
    to be a solid rectangle. Essentially, it recovers the correct image from
    a photo.
    """
    pic = pic.copy()
    mask = (pic[:,:,0]>100)*(pic[:,:,1]>100)*(pic[:,:,2]>100)

    larr, n_l = ndimage.label(mask)

    # get the label that makes up most of the center 50x50 pixels
    modeval = stats.mode(larr[int(pic.shape[0]/2.)-50:int(pic.shape[0]/2.)+50,int(pic.shape[1]/2.)-50:int(pic.shape[1]/2.)+50].ravel())[0][0]


    tmask = ndimage.binary_fill_holes(larr==modeval)

    # first get a bounding polygon. If it's not a quadrilateral, then fill in the missing regions in the mask until it is.
    approx = get_contours(tmask)

    if len(approx) > 4:
        # this will extend the lines of each side of the bounding polygon and fill in
        # regions where they intersect.
        tverts = [(ll[0][0],ll[0][1]) for ll in approx]
        tverts.append((approx[0][0][0], approx[0][0][1])) # append the first point to create the last line

        square_mask = np.zeros_like(tmask)
        for ii in range(len(tverts)-1):

            if abs(tverts[ii+1][0] - tverts[ii][0]) > abs(tverts[ii+1][1] - tverts[ii][1]):
                # horizontal line
                f = lambda x: (tverts[ii+1][1] - tverts[ii][1])/np.float((tverts[ii+1][0] - tverts[ii][0]))*(x - tverts[ii][0]) + tverts[ii][1]

                # there is definitely a faster way to do this but...
                for xii in range(square_mask.shape[1]):
                    square_mask[int(f(xii)),xii] = 1
            else:
                # vertical line
                f = lambda x: (tverts[ii+1][0] - tverts[ii][0])/np.float((tverts[ii+1][1] - tverts[ii][1]))*(x - tverts[ii][1]) + tverts[ii][0]

                # there is definitely a faster way to do this but...
                for xii in range(square_mask.shape[0]):
                    square_mask[xii,int(f(xii))] = 1


        filled_square = ndimage.binary_opening(ndimage.binary_fill_holes(square_mask))
        diffmask = np.logical_xor(tmask, filled_square)

        # replace the missing chunk with the mean of the tmask region
        for rgbii in range(3):
            pic[diffmask] = pic[tmask].mean()

        # the resulting "filled" mask should be a quadrilateral
        approx = get_contours(filled_square)


    # this calculates the aspect ratio to determine if the image is just the EKG
    # or if it's the larger scanned image
    props = measure.regionprops(tmask.astype(np.uint8))[0]

    aspect = props.major_axis_length/props.minor_axis_length
    if abs(aspect - 1.3) > abs(aspect - 1.65):
        # a scan or printout aspect ratio
        dst = np.array([[ll[0][0],ll[0][1]] for ll in approx])
        src = np.array([[(ll[0][0] > 4200/2.)*4200, (ll[0][1] > 2550/2.)*2550] for ll in approx])

        tform3 = tf.ProjectiveTransform()
        tform3.estimate(src, dst)
        warped = tf.warp(pic, tform3, output_shape=(2550,4200,3))
    else:
        # the ekg printout (not printed on 8.5x11 paper)
        dst = np.array([[ll[0][0],ll[0][1]] for ll in approx])
        src = np.array([[(ll[0][0] > 3310/2.)*3310, (ll[0][1] > 2530/2.)*2550] for ll in approx])

        tform3 = tf.ProjectiveTransform()
        tform3.estimate(src, dst)
        warped = tf.warp(pic, tform3, output_shape=(2530,3310,3))

    return warped, approx


def filter_pic(pic, size=10):
    """
    filter_pic applies a gaussian filter over the image with a size of 'size'

    We've sped this up by downsizing the image by a factor of 10, then taking the
    gaussian filter, then upscaling the results. It speeds things up by a factor
    of 20.
    """

    ttdown = tf.rescale(pic,.1)
    highpass = filters.gaussian(ttdown,size,mode='mirror')
    ttup = tf.resize(highpass, pic.shape[:2])

    filtimg_3 = pic - ttup

    # clip anything greater than zero, since these are from section with large
    # negative contrast, e.g. around the barcode
    filtimg_3[filtimg_3>0] = 0

    # and rescale so that min --> 0 and 0 --> 1
    for ii in range(3):
        rescale = np.abs(filtimg_3[:,:,ii].min())
        filtimg_3[:,:,ii] = (filtimg_3[:,:,ii] + rescale)/rescale

    return filtimg_3


def rotate_pic(pic):
    """
    rotate_pic finds the offset angle by measuring the grid with a Hough line transform. It then de-rotates by this amount
    """

    # for speed, only look at a 500x500 slice in the center of the image
    centercut = np.copy(pic[int(pic.shape[0]/2.)-500:int(pic.shape[0]/2.)+500,int(pic.shape[1]/2.)-500:int(pic.shape[1]/2.)+500,0])

    # find the grid in the image by looking for brighter regions
    centermask = (centercut<1)*(centercut>0.8)


    h, theta, d = tf.hough_line(centermask,theta=np.linspace(np.pi/2 - 0.2,np.pi/2 + 0.2,1000))
    _,angles,dist = tf.hough_line_peaks(h, theta, d)

    # plt.figure()
    # plt.imshow(centermask)
    # for ii in range(len(angles)):
    #     rho = dist[ii]
    #     theta = angles[ii]
    #
    #     y0 = (rho - 0 * np.cos(theta)) / np.sin(theta)
    #     y1 = (rho - centermask.shape[1] * np.cos(theta)) / np.sin(theta)
    #
    #     plt.plot([0,centermask.shape[0]],[y0,y1],'r-')

    angle = np.mean(angles*180/np.pi - 90)

    corrected_pic = tf.rotate(pic,angle,cval=1)

    return corrected_pic, angle


def get_ekg_segmentation(wfdb_path, model_path):
    """
    Gets segmentation data for ekg saved as wfdb.

    Note that thei format for each of qrs_segments, p_segments, t_segments is
    x_segments = [[left_endpoints],
                  [right_endpoints]]
    """
    eds = EcgDataset(path=wfdb_path, no_ext=True, sort=True)
    batch = (eds >> hmm_predict_pipeline(model_path, annot="hmm_annotation")).next_batch()
    full_meta = batch.data.meta[0]
    keep_fields = ['qrs',
                   'qt',
                   'pq',
                   'hr',
                   'qrs_segments',
                   'p_segments',
                   't_segments',
                   'recordname',
                  ]
    output_dict= { ff : full_meta[ff] for ff in keep_fields }
    return output_dict


class ProcessEKG:
    def __init__(self, maskdir, pic=None, isSmall=True, doTransform=True, doFilter=True, doRotate=True):
        self.__version__ = 'v05'

        masknames = ['aVR','aVL','aVF','V1','V2','V3','V4','V5','V6','I','II','III']
        masks = {}
        for mm in masknames:
            masks[mm] = np.load(maskdir + mm + '.npy')

        self._masks = masks
        self.leadnames = masknames
        self._isSmall = isSmall
        self._doTransform = doTransform
        self._doFilter = doFilter
        self._doRotate = doRotate
        if pic is not None:
            self._original_pic = pic
            self.pic = pic

    def load_pic(self, filepath, fobj=None):
        """
        load_pic loads the image located at 'filepath' into a numpy array. It is
        expecting a NxMx3 image (e.g. PNG or JPG).
        """

        if fobj is not None:
            self._filepath = None
            self._imagename = 'no_name'
            pic = fobj
        else:
            self._filepath = filepath
            self._imagename = filepath.split('/')[-1]
            pic = plt.imread(filepath)
            
        self._original_pic = pic
        # if np.max(pic) > 1:
        #     pic = pic/255.
        self.pic = pic


    def crop_pic(self):
        """
        crop_pic isolates the chart region (which contains the waveforms)

        This adds the cropped regions to the object as _cut_chart. It also
        adds _allind which is a dictionary containing
        the x,y coordinates of the top left and bottom right corners of the bounding
        boxes for the region.

        Because the EKG charts are the same size, this has hard-coded crop
        values based on the dimensions of the image. This will be changed
        in future versions. Note that in previous versions, cropping was dependent
        on found features of the image, but this was changed when we started using
        photographs of the image instead of scanned images.


        isSmall - whether the image is the small EKG printout or a larger 8.5x11
                    sheet of paper.
        """

        if self._isSmall:
            self._allind = {'chart': [(740,240),(2400,3250)]}

            self._cut_chart = self.pic[740:2400,240:3250,:]
        else:
            # x: 1100:4130
            # y: 716:2380

            self._allind = {'chart': [(716,1100),(2380,4130)]}

            self._cut_chart = self.pic[716:2380,1100:4130,:]



    def crop_chart(self, thresh=0.7, spacing=100, border=10, numpeaks=4):
        """
        thresh is used to crop the signal from the background. The wafeforms
        appear as dark lines, usually between 0 and 0.2 (0 being black, 1 being white)
        for scanned images and less than 0.7 for the filtered images.

        spacing, border, and numpeaks are parameters passed to find_peaks. Check the
        find_peaks doc for more info.
        """
        mask = (self._cut_chart[:,:,0] < thresh) * (self._cut_chart[:,:,1] < thresh) * (self._cut_chart[:,:,2] < thresh)

        # find where the timestreams are
        # this is done by summing the X axis values of the mask
        masksum = mask.sum(axis=1)

        # we want to get the extent of the signal for each timestream
        # either fit a gaussian and use the width, or a peak-finder and the baseline

        peakind = find_peaks(masksum, spacing, border, numpeaks)
        peakind.sort()


        self._peakind = peakind
        self._xcut = self._cut_chart[:int(peakind[-2] + (peakind[-1] - peakind[-2])/2),:,:]
        self._heartrate_cut = self._cut_chart[int(peakind[-2] + (peakind[-1] - peakind[-2])/2):,:,:]


    def isolate_leads(self, leads_per_row=4):
        """
        leads_per_row is the number of leads in each row. Since there are 12
        leads total, these are usually broken up into 4 leads per each row
        and 3 rows.
        """
        masks = self._masks
        self._leads_per_row = leads_per_row

        maxLocs = {}
        for mm in masks:
            # this finds each text mask template (e.g. 'I', 'aVL', 'V1', etc.)
            # and gives the location in the xcut cropped image
            results = cv2.matchTemplate(1-self._xcut[:,:,0].astype(np.float32),masks[mm].astype(np.float32), cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(results)
            maxLocs[mm] = max_loc


        # find any misplaced ones
        allx = []
        ally = []
        for mm in masks:
            allx.append(maxLocs[mm][0])
            ally.append(maxLocs[mm][1])

        allx = np.asarray(allx)
        ally = np.asarray(ally)

        # if it's by itself in x or y, pop from list
        # else if it's xdiff or ydiff is not within 10 of 720 or 400 of the nearest neighbor pop it
        problemMask = []
        for mm in masks:
            tx = maxLocs[mm][0]
            ty = maxLocs[mm][1]

            if np.sum(np.abs((allx-tx))<10) == 1:
                problemMask.append(mm)
            elif np.sum(np.abs((ally-ty))<10) == 1:
                problemMask.append(mm)
            # these next ones will determine if it's within a cluster of badly recognized images
            # by seeing if it has any neighbors that are 730 +/ 15 and 406 +/- 15 pixels away
            # COMMENTED OUT FOR NOW since it's not an observed issue and it's not doing exactly what I want.
            # elif np.sum((np.abs(np.abs(allx-tx) - 730) < 15) * (np.abs(np.abs(ally-ty) - 406) < 15)) < 1:
            #     problemMask.append(mm)

        # these two objects show the ordering of the leads in the 12-lead EKG
        # WARNING: if the EKG type changes, these will not work
        maskPos = {'I': (0,0), 'II': (0,1), 'III': (0,2),
                   'aVR':(1,0), 'aVL': (1,1), 'aVF': (1,2),
                   'V1': (2,0), 'V2': (2,1), 'V3': (2,2),
                   'V4': (3,0), 'V5': (3,1), 'V6': (3,2)}

        invMaskPos = np.array([['I','II','III'],['aVR','aVL','aVF'],['V1','V2','V3'],['V4','V5','V6']],dtype=str)

        self._maskPos = maskPos
        self._invMaskPos = invMaskPos

        # replace missing lead
        # this is ugly but it gets the job done
        for mm in problemMask:
            tpos = maskPos[mm]
            xii = tpos[0]
            yii = tpos[1]

            notFound = True
            goodneighbor = None
            xtry = 0
            ytry = 0
            lastone = 0
            while notFound and xtry < 4 and ytry < 3:
                # look over one in x if it's also missing
                # then look over one in y
                # if those are missing, look over 2 in x
                # if that's missing, look over 2 in y
                # keep going until it finds one

                if invMaskPos[(xii+xtry)%4,(yii+ytry)%3] not in problemMask:
                    goodneighbor = invMaskPos[(xii+xtry)%4,(yii+ytry)%3]
                    notFound = False
                else:
                    if lastone == 1:
                        xtry += 1
                        lastone = 0
                    else:
                        ytry += 1
                        lastone = 1

            # look for the max value near the area determined from the closest neighbor
            results = cv2.matchTemplate(1-self._xcut[:,:,0].astype(np.float32),masks[mm].astype(np.float32), cv2.TM_CCOEFF)
            ystart = max([0, maxLocs[goodneighbor][0] - 730*(xii - maskPos[goodneighbor][0])-50])
            xstart = max([0, maxLocs[goodneighbor][1] + 406*(yii - maskPos[goodneighbor][1])-50])
            # note that the -/+ for x and y are different because the image has x-increasing from top to bottom
            yend = min([results.shape[0], ystart + 100])
            xend = min([results.shape[1], xstart + 100])

            tresults = np.zeros_like(results)
            tresults[xstart:xend,ystart:yend] = results[xstart:xend,ystart:yend]

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(tresults)

            maxLocs[mm] = max_loc



        # this section crops the regions around each of the 12 leads
        # the crop region is defined by the center lines (as found in peakind)
        # and the spacing between the leads as found by the template matching above
        # e.g. in maxLocs
        # this returns the cropped regions for each of the 12 leads and the
        # location of the text in each cropped region
        xspace = int((max([maxLocs[mm][0] for mm in maxLocs]) - min([maxLocs[mm][0] for mm in maxLocs]))/np.float(leads_per_row-1))
        yspace = np.mean(np.diff(self._peakind)/2).astype(np.int)
        crops = {}
        text_coords = {}
        for mm in masks:
            ystart = self._peakind[np.argmax(-np.abs(maxLocs[mm][1] - self._peakind))]
            crops[mm] = self._xcut[max([ystart - yspace,0]):ystart + yspace,maxLocs[mm][0]:maxLocs[mm][0]+xspace]

            # locations of the text within the crop
            text_coords[mm] = maxLocs[mm][1] - max([ystart - yspace,0])

        self._crops = crops
        self._text_coords = text_coords
        self._maxLocs = maxLocs

    def isolate_heartrate(self):
        """
        similar to isolate_leads, isolate_heartrate finds the location of the
        text mask for the heartrate_cut image. It loops through all the masks
        to find the best-fit. In some cases, II will be the best fit for I and
        III will be the best fit for II.
        This isn't a problem since the goal is to subtract the text.
        Subtracting excess area around the text is fine.

        The result of this function is it will add a 'heartrate' key to _maxLocs
        _masks _crops and _text_coords.
        """
        masks = self._masks

        maxLocs = {}
        maxVals = {}
        for mm in masks:
            # this finds each text mask template (e.g. 'I', 'aVL', 'V1', etc.)
            # and gives the location in the xcut cropped image
            results = cv2.matchTemplate(1-self._heartrate_cut[:200,:200,0].astype(np.float32),masks[mm].astype(np.float32), cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(results)
            maxLocs[mm] = max_loc
            maxVals[mm] = max_val

        tmax = 0
        for mm in maxVals:
            if maxVals[mm] > tmax:
                tmax = maxVals[mm]
                lmax = mm

        self._crops['heartrate'] = self._heartrate_cut
        self._maxLocs['heartrate'] = maxLocs[lmax]
        self._masks['heartrate'] = masks[lmax]
        self._text_coords['heartrate'] = maxLocs[lmax][1]




    def get_waveform(self, leadname, ithresh=0.2, border=10, blobsize=20, nearest_neighbor=20, growth_iter=10):
        """
        border ignores anything within 'border' pixels from the top, bottom, and
        right side of the image. This is used to reduce artifacts or bleed-over
        from other nearby cropped regions. NOTE: this may no longer be necessary
        since improving the cleaning algorithms.

        blobsize sets the minimum size for any contiguous region in the cropped
        pic. It's used to remove the grid dots from the thresholded binary mask.

        nearest_neighbor sets the distance scale to consider two regions part of
        the same waveform. This is used to remove artifacts that pass the thresholding
        and cleaning algorithms, but are clearly not the signal (e.g. creases in the
        printout, cuts, holes, etc.).

        growth_iter also deals with removing artifacts. Candidate regions are
        dilated by 'growth_iter' iterations to look for regions that are
        close but were not touching. nearest_neighbor and growth_iter should
        be tuned together.
        """
        mm = leadname

        tt = np.copy(self._crops[mm][:,:,0])

        # remove the text
        ttext = np.zeros(tt.shape, dtype=np.uint8)
        if mm == 'heartrate':
            # the heartrate lead isn't cropped in the same way as the others. It needs the maxLocs x and y positions
            ttext[self._maxLocs[mm][1]:self._maxLocs[mm][1]+self._masks[mm].shape[0],self._maxLocs[mm][0]:self._maxLocs[mm][0]+self._masks[mm].shape[1]] = self._masks[mm]
        else:
            ttext[self._text_coords[mm]:self._text_coords[mm]+self._masks[mm].shape[0],0:self._masks[mm].shape[1]] = self._masks[mm]

        # dilate to remove the edges too
        ttext = ndimage.binary_dilation(ttext,iterations=2)

        tt[ttext==1] = 1

        # subtract a border from the top, bottom, and right side
        tt[:border,:] = 1
        tt[-border:,:] = 1
        tt[:,:border] = 1
        tt[:,-border:] = 1

        ithresh = 0.2
        cleaned = remove_spots(tt<ithresh,40)
        while len(np.where(np.sum(cleaned,axis=0)==0)[0])/np.float(cleaned.shape[1]) > 0.05 and ithresh<1:
            ithresh += 0.02

            cleaned = remove_spots(tt<ithresh,40)

        # remove any larger regions that make it past the blob cut
        larr, n_ = ndimage.label(ndimage.binary_dilation(cleaned,iterations=12))
        if n_ > 1:
            idxcenter = ndimage.center_of_mass(larr)[0]
            tcenter = np.zeros_like(larr)
            tcenter[int(idxcenter)-10:int(idxcenter)+10] = 1

            tokeep = np.zeros_like(larr)
            for ii in np.unique(larr*tcenter)[1:]:
                tokeep += (larr == ii)

            cleaned = cleaned * tokeep

        signal = extract_waveform(cleaned)

        return cleaned, signal

    def get_waveforms(self):
        """
        This is just a wrapper around get_waveform which applies the waveform
        extraction to each of the leads in the image. It add the cleaned
        2-D signal binary masks and the extracted waveforms from these masks to
        the object as _signal_masks and waveforms.
        """
        waveforms = {}
        signal_masks = {}
        for mm in self.leadnames:
            signal_masks[mm], waveforms[mm] = self.get_waveform(mm)

        self._signal_masks = signal_masks
        self.waveforms = waveforms

    def get_heartrate(self):
        signal_mask, waveform = self.get_waveform('heartrate')

        self._signal_masks['heartrate'] = signal_mask
        self.waveforms['heartrate'] = waveform

    def save_results(self, dosave=False, savedir=None, savename=None, format=None):
        outdict = {}

        return outdict


    def plot_ekg(self, dosave=False, savepath=''):
        pic = self.pic
        maxLocs = self._maxLocs
        peakind = self._peakind

        fig,ax = plt.subplots()
        ax.imshow(pic)
        #overlay = np.zeros((pic.shape[0],pic.shape[1]))
        #overlay[overlay==0] = np.nan
        xspace = int((max([maxLocs[mm][0] for mm in maxLocs]) - min([maxLocs[mm][0] for mm in maxLocs]))/np.float(self._leads_per_row-1))
        yspace = np.mean(np.diff(self._peakind)/2).astype(np.int)
        for mm in self.leadnames:
            startcoords = (maxLocs[mm][0] + self._allind['chart'][0][0], maxLocs[mm][1]+self._allind['chart'][0][1])
            rect = patches.Rectangle(startcoords,self._masks[mm].shape[1],self._masks[mm].shape[0],linewidth=1,edgecolor='r',facecolor='none')

            ax.add_patch(rect)

            ystart = self._peakind[np.argmax(-np.abs(maxLocs[mm][1] - peakind))] + self._allind['chart'][0][1]
            outline = patches.Rectangle((startcoords[0],ystart - yspace),xspace,2*yspace,linewidth=1,edgecolor='c',facecolor='none')

            ax.add_patch(outline)

            X = np.arange(self.waveforms[mm].shape[0])
            plt.plot(X + startcoords[0], self.waveforms[mm] + startcoords[1] - maxLocs['I'][1],'r')


            #overlay[ystart - yspace:ystart - yspace+signal_masks[mm].shape[0],
            #        startcoords[0]:startcoords[0]+signal_masks[mm].shape[1]] = signal_masks[mm]

        #plt.imshow(overlay,alpha=0.6)
        #plt.clim([0,1.3])
        plt.show()
        if dosave:
            savename = self._imagename.replace('-rot','-rot_segmented')
            plt.savefig(savepath + version + '/' + savename)

    def process_image(self, filepath):
        """
        This runs through the necessary steps to extract all waveforms from an
        ekg image.

        Use plot_ekg after this to visualize.
        """
        self.load_pic(filepath)

        if self._doTransform:
            transformedpic, approx = transform_pic(self.pic)
            self._approx = approx
            self.pic = transformedpic
        if self._doFilter:
            self.pic = filter_pic(self.pic)

        if self._doRotate:
            self.pic, self._angle = rotate_pic(self.pic)

        self.crop_pic()
        if not self._isSmall and not self._doFilter:
            self.crop_chart(thresh=0.2)
        else:
            self.crop_chart()
        self.isolate_leads()
        self.get_waveforms()

        # get heartrate lead
        self.isolate_heartrate()
        self.get_heartrate()

    def _nan_helper(self, y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        """
        return np.isnan(y), lambda z: z.nonzero()[0]


    def wfdict_to_wfdb(self, wfdict=None, filename=None, fs=296, splitleads=True, savepath=None):
        """
        Convert dictionary of 12-lead waveforms to wfdb format for use in CardIO

        INPUT:

            wfdict (dict, optional):
                dictionary of waveforms as numpy 1-D arrays which have been
                cleaned and interpolated. If not supplied, it will use the
                waveforms stored in this object.
            filename (str, required):
                name to be used for wfdb files with no extension
            fs (int/float, optional):
                Sampling frequency for the EKG, calculated from EKGs provided by
                MBR.
            splitleads (bool, optional):

        OUTPUT:
            Does not return a value, but it saves .dat and .hea files in the same
            folder as the original datafiles.
        """
        if wfdict:
            self.wf_odict = OrderedDict(wfdict)
        elif self.waveforms:
            self.wf_odict = OrderedDict(self.waveforms)
        else:
            raise AttributeError('No waveforms to be written to wfdb.')

        # The code below pre-processes and then saves the waveforms as a wfdb.
        # This includes subtracting the mode (used as a baseline), replacing
        # NaNs with an interpolation.
        for key, val in self.wf_odict.items():
            # Remove baseline
            val = np.array(val)
            val -= stats.mode(val)[0][0]

            # Interpolate over NaNs
            nans, x = self._nan_helper(val)
            val[nans] = np.interp(x(nans), x(~nans), val[~nans])
            val = val.reshape(-1, 1)
            self.wf_odict[key] = val

            # Define some necessary metadata
            try:
                record_name = filename[:-4] + '-' + str(key)
                units = ['mV']
            except:
                print('Need to supply a filename for ekg to convert.')

            # Save
            print('Saving {}'.format(record_name))
            wrsamp(
                recordname=record_name,
                fs=fs,
                units=units,
                signames=[str(key)],
                fmt=['80'],
                p_signals=np.array(val)
            )
            if savepath:
                self._savepath = savepath
                headats = list(Path('.').glob('*' + str(record_name) + '*.hea'))
                headats += list(Path('.').glob('*' + str(record_name) + '*.dat'))
                for hd in headats:
                    #try:
                    shutil.move(
                        str(hd),
                        os.path.join(
                            str(self._savepath),
                            str(hd.name)
                        )
                    )
                    #except:
                    #    pass


# import time
#
# t1 = time.time()
# EKG = ProcessEKG('/FIR/MBR/EKG/iOS_backend/ecg_ios/static/text_masks/',isSmall=True, doTransform=True, doFilter=True, doRotate=True)
# EKG.process_image('/FIR/MBR/EKG/data/photos/test_photo_02.jpg')
#
# print(time.time() - t1)
#
