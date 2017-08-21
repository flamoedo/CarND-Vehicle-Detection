# ## Detection Pipeline

# ### Auxiliar Functions

# In[130]:

import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from auxiliar_functions import convert_color, get_hog_features, bin_spatial, color_hist

# ### Detection

# In[151]:
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    if invert_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    global spatial_features 
    global hist_features 
    global hog_features
    
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)

    # ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    bbox = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            #hog_features = hog_feat1
            

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (32,32))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
                bbox.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return bbox, draw_img
    


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    
    global aheat, hframes

    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    
    #hold the last 5 frames to stabilize detection
    aheat.append(heatmap)
    
    aheat = aheat[-hframes:]
    
    nheat = np.mean(aheat, axis=0)
    
    return nheat

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        if invert_rgb:
            cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 2)
        else:
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 2)

    # Return the image
    return img


def pipeline(img):

    global threshold
    
    out_bbox, out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    box_list = out_bbox
    
    # Read in image similar to one shown above 
    # image = mpimg.imread('test_image.jpg')
    image = img
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)	

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,threshold)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255).astype(np.uint8)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    return draw_img


def process_video(video_path, write_output):
    
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    
    global invert_rgb
    
    set_global_variables()
    
    invert_rgb = False
    
    #Extend lines in proportion to image
    IMAGE_PROPORTION = True
    
    clip1 = VideoFileClip(video_path)
    white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(write_output, audio=False)
    
def set_global_variables():
    
    global dist_pickle,svc,X_scaler ,orient ,pix_per_cell ,cell_per_block ,\
    spatial_size,hist_bins ,ystart,ystop ,scale, aheat, threshold, hframes, invert_rgb
    
    dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    threshold = 0
    hframes = 60

    ystart = 400
    ystop = 656
    scale = 1.5
    
    aheat = []
    

    
def run_video(video_path):

    global invert_rgb

    set_global_variables()
    
    invert_rgb = True
    
    cap = cv2.VideoCapture(video_path)
    
    ret = True    
        
    ret, img = cap.read()
    
    
    while(ret):
        # Capture frame-by-frame
        
        ret, img = cap.read()
                
        img_draw = pipeline(img)
        
        cv2.imshow('frame', img_draw)       
        
        #thresh1 = cv2.applyColorMap(heatmap,cv2.COLORMAP_HOT)
        
        #print(heatmap.shape)
        
        #cv2.imshow('frame2', thresh1 * 20)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
if __name__ == "__main__":
    #run_video('project_video.mp4')
    
    #run_video('test_video.mp4')
    
    process_video('project_video.mp4', 'project_video_output3.mp4')
    
    #run_video('challenge_video.mp4')
    
    #run_video('harder_challenge_video.mp4')





