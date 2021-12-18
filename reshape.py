import cv2
import numpy as np
from numpy.core.fromnumeric import size 
#if saving intensity graphs:
import matplotlib as mpl

import matplotlib.pyplot as plt
from scipy.signal import argrelextrema  
#from scipy.signal.ltisys import StateSpace, TransferFunctionDiscrete

show_steps = False
save_plots = False
path = "Figures/Bilder/resultater/as-grown/p320_11/"
orig = path[-8:-1]+".png"
stretch = float(path[-3:-1])/10
print(stretch)
horizontal = True
vertical = True
manual_override = False

light_side_left = True
light_side_top = True
#Disse har ingenting for seg, er alltid True, True uansett. Kan egt fjernes. 
defined_top = True
defined_left = True

#crop parameters   
crop_blur = 137 #higher val means more blur for crop, value must be odd, f.eks. 137
crop_thresh = [161, 3] #first value defines neighborhood size  for adaptive threshold, second value is for fine tuning, f.eks. [261, 7]
thresh_area = 6000 #deletes contours smaller than thresh_ area pixels. f.eks. 6000
crop_x =   0 #adds extra in crop for specified direction. 
crop_y = 0

#scale params
edge_threshold = 0  #how far from edges do we ignore peaks     
#round_thresh = 0.02 #how close to exactly n*diff_avg does value need to be to add extra point
#atm the two next values are specified manually, but could be specified automatically using some easy statistics (range, median, mean etc. )
min_diff =  14 #how close points are automatically removed
mid_diff =  60 #how close do points have to be on opposite sides for point to be removed (to remove unwanted peaks between the wanted peaks). This should be slightly above diff_avg/2. 
#x-scale params  
x_blur = [71, 137] #higher values gives more blur, blur defined in both x and y directions. Both must be odd. These should be tuned to find the correct points. 
#y-scale params
y_blur = [77, 57] 

if save_plots: 
    print("Can not show steps because plots are being saved as vector graphics")
    show_steps = False
    mpl.use("svg")
    new_rc_params = {'text.usetex': False, "svg.fonttype": 'none', 'figure.figsize':(6, 2)
    }
    mpl.rcParams.update(new_rc_params)  
    f, (ax1, ax2) = plt.subplots(1,2)
#read image
image  = cv2.imread(path + orig)
if show_steps:
    cv2.imshow("Image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

#make grayscale 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#gaussion blur, image, kernel  size (height, width, should be odd), feks 137
blur = cv2.GaussianBlur(gray, (crop_blur,crop_blur), 0)
if show_steps:
    cv2.imshow("blur", blur)
    cv2.waitKey()
    cv2.destroyAllWindows()

#threshold image
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, crop_thresh[0], crop_thresh[1] ) #last numbers changes thresholding somewhat, approx 261, 7
if show_steps:
    cv2.imshow("thresh", thresh)
    cv2.waitKey()
    cv2.destroyAllWindows()

#extract all contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#find contours with area larger than thresh_area
c = 0
cnt = [] 
mask = np.zeros((gray.shape),np.uint8)
for i in contours:
        area = cv2.contourArea(i)
        if area > thresh_area:
            cnt.append(i)
            mask = cv2.drawContours(mask, contours, c, (0, 255, 0), 3)
        c+=1

#create mask of largest contours
cv2.drawContours(mask,cnt,-1,255,-1)
cv2.drawContours(mask,cnt,-1,0,2)
if show_steps:
    cv2.imshow("mask", mask)
    cv2.waitKey()
    cv2.destroyAllWindows()

rect = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
x,y,w,h = cv2.boundingRect(mask)
cv2.rectangle(rect,(x, y),(x+w,y+h),(55,98,226),30)
if show_steps:
    cv2.imshow("rectangle", rect)
    cv2.waitKey()
    cv2.destroyAllWindows()
cv2.imwrite(path + "rect.png", rect)


#crop
frame = 50 #The MFM-images often get a border which is picked up by the contours, this is to compensate for some of that before the overlay is added. 
#cropped_image = cv2.cvtColor(image[y + frame:y+h-frame, x+frame:x+w-frame], cv2.COLOR_BGR2GRAY) #crop with frame
cropped_image = cv2.cvtColor(image[y-crop_y:y+h+crop_y, x-crop_x:x+w+crop_x], cv2.COLOR_BGR2GRAY) #crop based on square
#cropped_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #no crop
if show_steps:
    cv2.imshow("cropped", cropped_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

#-------------
#resize in x

#blur
cropped_blur = cv2.GaussianBlur(cropped_image, (x_blur[0],x_blur[1]), 0) #if more than 26 peaks or valleys are found, increase blur
 
#Find intensity in lines down entire image
height = len(cropped_blur)
width = len(cropped_blur[0])
x = np.linspace(0, width, width)
val = [0]*width 
for row in cropped_blur:
    val += row
#val is the sum of intensities in whole image

#Find local maxima in intensity
max = argrelextrema(val, np.less)[0] #np.greater if we wish to use the lightest points, max is the local minima NB!!! If we choose to use lightest poitns, light_side_left/top must be opposite of what is true.
diff = [j-i for i, j in zip(max[:-1], max[1:])]
diff_avg = np.average(diff)
print(max)
print(diff)
if max[0]<edge_threshold:
    max = max[1:]
if max[-1]>(width-edge_threshold):
    max = max[:-1]
remove = []
for i in range(len(diff)):
    if diff[i]<min_diff:
        remove.append(i)
max = [max[i] for i in range(len(max)) if i not in remove]
diff = [j-i for i, j in zip(max[:-1], max[1:])]
remove = []
for i in range(len(diff)-1):
    if (diff[i]<mid_diff) and (diff[i+1]<mid_diff): 
        remove.append(i+1)
max = [max[i] for i in range(len(max)) if i not in remove]
diff = [j-i for i, j in zip(max[:-1], max[1:])]
print(diff)
print(len(max))
if show_steps:
    plt.plot(x, val)
    plt.plot(max, val[max], '.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
diff_avg = np.average(np.sort(diff)[2:-2])
print(diff_avg)
autoadd = np.array([], dtype = int)
if len(max) < 26:
    for i in range(len(diff)-1, -1, -1):
        extra = round(diff[i]/diff_avg)-1 
        if extra>0:
            space = diff[i]/(extra+1)
            for j in range(extra):
                #max = np.append(max, int(max[i]+(j+1)*diff_avg))
                max = np.append(max, int(max[i]+(j+1)*space))
                autoadd = np.append(autoadd, int(max[i]+(j+1)*space))
print(autoadd)
max = np.unique(max)
print(len(max))
print(max)
max = np.sort(max)
if save_plots:
    ax1.plot(x/1000, val, color = '#3d66af', linewidth = 1)
    ax1.plot(max/1000, val[max], '.', color = '#e26237', markersize = 4)
    ax1.plot(autoadd/1000, val[autoadd], '.', color = '#621743', markersize = 4)
    ax1.set_yticklabels([])
    ax1.set_xlabel("x [1000 pixels]\n\n(a)", labelpad = 10) 
    ax1.set_ylabel("Intensity", labelpad = 10)
    ax1.grid()
if show_steps:
    plt.plot(x, val)
    plt.plot(max, val[max], '.')
    plt.xlabel("x [pixels]") 
    plt.ylabel("Total intensity")
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
if manual_override:
    deleted_x = []
    inp = input('delete? (y/n)')
    while inp == 'y': 
        ind = int(input('index:'))
        deleted_x.append(ind)
        max = np.delete(max, ind)
        inp = input('delete? (y/n)')
    added_x = []
    inp = input('add? (y/n)')
    while inp == 'y':
        ind = int(input('x value:'))
        added_x.append(ind)
        max = np.append(max, ind)
        inp = input('add? (y/n)')
max = np.sort(max)
if show_steps and manual_override:
    plt.plot(x, val)
    plt.plot(max, val[max], '.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()


resize = cropped_image.copy()
if vertical: 
    resize = resize[:height, :max[0]]
    if light_side_left:
        resize = cv2.resize(resize, [int(diff_avg/2), height])
    else: 
        resize = cv2.resize(resize, [1, height])
    if show_steps:
        cv2.imshow("final resize", resize)
        cv2.waitKey()
        cv2.destroyAllWindows()
    parts = []
    for i in range(1, 26):
        part = cropped_image[:height, max[i-1]:max[i]]
        part = cv2.resize(part, [int(diff_avg), height])
        resize = cv2.hconcat([resize, part])
        if show_steps:
            cv2.imshow("part %s"%i, resize)
            cv2.waitKey()  
            cv2.destroyAllWindows()
    end = cropped_image[:height, max[-1]:]
    if light_side_left:
        end = cv2.resize(end, [1, height])
    else:
        end = cv2.resize(end, [int(diff_avg/2), height])
    resize = cv2.hconcat([resize, end])
    if show_steps:
        cv2.imshow("x-resize", resize)
        cv2.waitKey()
        cv2.destroyAllWindows()
else: 
    if defined_left:
        if light_side_left:
            beg = resize[:height, :max[0]]
            resize = resize[:height, max[0]:max[-1]]
            beg = cv2.resize(beg, [int(len(resize[0])/(2*25)), len(beg)]) #DETTE MÅ OPPDATERES I RESTEN AV KODEN OG TING MÅ KJØRES PÅ NYTT
            resize = cv2.hconcat([beg, resize])
        else: 
            end = resize[:height, max[-1]:]
            resize = resize[:height, max[0]:max[-1] ]
            end = cv2.resize(end, [int(len(resize[0])/(2*25)), len(end)])
            resize = cv2.hconcat([resize, end])
            print('crop')

resize = cv2.resize(resize, [int(stretch*height), height])
if show_steps:
    cv2.imshow("resize", resize)
    cv2.waitKey()
    cv2.destroyAllWindows()

max_x = max
cropped_image = resize.copy()
print('x-resize done.')
#-------------
#resize in y
#blur
cropped_blur_x = cv2.GaussianBlur(resize, (y_blur[0],y_blur[1]), 0)
#Find intensity in lines down entire image
height = len(cropped_blur_x)
width = len(cropped_blur_x[0])
val = [0]*height
for y in range(height):
    for x in range(width): 
        val[y] += cropped_blur_x[y][x]
#Find local maxima in intensity
y = np.linspace(0, height, height)
val = np.array(val)
max = argrelextrema(val, np.less)[0] #np.greater if we wish to use the lightest points, max is the local minima NB!!! If we choose to use lightest poitns, light_side_left/top must be opposite of what is true.
diff = [j-i for i, j in zip(max[:-1], max[1:])]
diff_avg = np.average(diff)
if max[0]<edge_threshold:
    max = max[1:]
if max[-1]>(height-edge_threshold):
    max = max[:-1]
remove = []
for i in range(len(diff)):
    if diff[i]<min_diff:
        remove.append(i)
max = [max[i] for i in range(len(max)) if i not in remove]
diff = [j-i for i, j in zip(max[:-1], max[1:])]
remove = []
for i in range(len(diff)-1):
    if (diff[i]<mid_diff) and (diff[i+1]<mid_diff): 
        remove.append(i+1)
max = [max[i] for i in range(len(max)) if i not in remove]
diff = [j-i for i, j in zip(max[:-1], max[1:])]
print(len(max))
print(max) 
if show_steps:
    plt.plot(y, val)
    plt.plot(max, val[max], '.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
diff_avg = np.average(np.sort(diff)[2:-2])
print(diff)
print(diff_avg)
autoadd = np.array([], dtype = int)
if len(max) < 26:
    for i in range(len(diff)-1, -1, -1):
        extra = round(diff[i]/diff_avg)-1 
        if extra>0:
            space = diff[i]/(extra+1)
            for j in range(extra):
                #max = np.append(max, int(max[i]+(j+1)*diff_avg))
                max = np.append(max, int(max[i]+(j+1)*space))
                autoadd = np.append(autoadd, int(max[i]+(j+1)*space))
max = np.unique(max)
print(len(max))
print(max)
max = np.sort(max)
if save_plots:
    ax2.plot(y/1000, val, color = '#3d66af', linewidth = 1)
    ax2.plot(max/1000, val[max], '.', color = '#e26237', markersize = 4)
    ax2.plot(autoadd/1000, val[autoadd], '.', color = '#621743', markersize = 4)
    ax2.set_yticklabels([])
    ax2.set_xlabel("y [1000 pixels]\n\n(b)", labelpad = 10)
    ax2.grid()
    f.savefig(path + "reshape.svg", bbox_inches = "tight")
if show_steps:
    plt.plot(y, val)
    plt.plot(max, val[max], '.')
    plt.xlabel("y [pixels]")
    plt.ylabel(path + "Total intensity")
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
if manual_override:
    deleted_y = []
    inp = input('delete? (y/n)')
    while inp == 'y': 
        ind = int(input('index:'))
        deleted_y.append(ind)
        max = np.delete(max, ind)
        inp = input('delete? (y/n)')
    added_y = []
    inp = input('add? (y/n)')
    while inp == 'y':
        ind = int(input('x value:'))
        added_y.append(ind)
        max = np.append(max, ind)
        inp = input('add? (y/n)')
max = np.sort(max)
if show_steps and manual_override:
    plt.plot(y, val)
    plt.plot(max, val[max], '.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

if horizontal: 
    resize = resize[:max[0],]
    if light_side_top:
        resize = cv2.resize(resize, [width, int(diff_avg/2)])
    else: 
        resize = cv2.resize(resize, [width, 1])
    if show_steps:
        cv2.imshow("final resize", resize)
        cv2.waitKey()
        cv2.destroyAllWindows()
    parts = []
    for i in range(1, 26):
        part = cropped_image[max[i-1]:max[i],]
        part = cv2.resize(part, [width, int(diff_avg)])
        resize = cv2.vconcat([resize, part])
        if show_steps:
            cv2.imshow("part %s"%i, resize)
            cv2.waitKey()  
            cv2.destroyAllWindows()
    end = cropped_image[max[-1]:,]
    if light_side_top:
        end = cv2.resize(end, [width, 1])
    else:
        end = cv2.resize(end, [width, int(diff_avg/2)])
    resize = cv2.vconcat([resize, end])
    if show_steps:
        cv2.imshow("y-resize", resize)
        cv2.waitKey()
        cv2.destroyAllWindows()
else: 
    if defined_top:
        if light_side_top:
            beg = resize[:max[0],]
            resize = resize[max[0]:max[-1],]
            beg = cv2.resize(beg, [len(beg[0]), int(len(resize)/(2*25))])
            resize = cv2.vconcat([beg, resize])
        else: 
            end = resize[max[-1]:, ]
            resize = resize[max[0]:max[-1], ]
            end = cv2.resize(end, [len(end[0]), int(len(resize)/(2*25))])
            resize = cv2.vconcat([resize, end])
            print('crop')

resize = cv2.resize(resize, [int(stretch*height), height])
if show_steps:
    cv2.imshow("resize", resize)
    cv2.waitKey()
    cv2.destroyAllWindows()
print('y-resize done.')

cv2.imwrite(path + "resize.png", resize)


#-----------
#Write parameters to 'params.text'-file

f = open(path + 'params.txt', 'w')
f.write('original file: \t' + path + orig + '\n')
f.write('stretch: \t %s \n\n'%stretch)
f.write('horizontal: \t %s\n'%horizontal)
f.write('vertical: \t %s\n\n'%vertical)
f.write('light_side_left: \t %s\n'%light_side_left)
f.write('light_side_top: \t %s\n'%light_side_top)
f.write('defined_left: \t %s\n'%defined_left)
f.write('defined_top: \t %s\n\n'%defined_left)
f.write('Crop parameters:\n')
f.write('crop_blur: \t %s\n'%crop_blur)
f.write('crop_thresh: \t %s\n'%crop_thresh)
f.write('thresh_area: \t %s\n'%thresh_area)
f.write('crop_x:\t%s\n'%crop_x)
f.write('crop_y:\t%s\n\n'%crop_y)
f.write('Scale parameters:\n')
f.write('edge_threshold: \t%s\n'%edge_threshold)
#f.write('round_thresh: \t %s\n'%round_thresh)
f.write('min_diff: \t %s\n'%min_diff)
f.write('mid_diff:\t %s\n'%mid_diff)
f.write('x_blur:\t%s\n'%x_blur)
f.write('y_blur:\t%s\n\n'%y_blur)
f.write('manual_override:\t%s\n'%manual_override)
if manual_override:
    f.write('deleted_x:\t%s\n'%deleted_x)
    f.write('added_x:\t%s\n'%added_x)
    f.write('deleted_y:\t%s\n'%deleted_y)
    f.write('added_y:\t%s\n'%added_y)
f.write('\n max_x:\t%s\n'%max_x)
f.write('max_y:\t%s\n'%max)
f.close()