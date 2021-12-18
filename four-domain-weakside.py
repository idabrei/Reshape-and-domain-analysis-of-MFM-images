import cv2
import numpy as np
from numpy.core.fromnumeric import size 

show_steps = True
path = "Figures/Bilder/resultater/phi45-2-b100-rt/p340_08/"
stretch = float(path[-3:-1])/10
threshold = 0
flux = 'right' #true for vertical weak direction, false for horizontal 
#fikse litt på den siste delen der, kanskje spesifisere up/down, right/left og passe på at det lengre nede i koden stemmer med det

#Read reshaped image and make grayscale
image  = cv2.imread(path + "resize.png")
image = cv2.resize(image, (int(image.shape[1]/6), int(image.shape[0]/6)))
height = len(image)
if show_steps:
    cv2.imshow("Reshaped", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#gray = cv2.GaussianBlur(image, (37, 37), 0)

#---------
#create pattern
#read image
pattern_og  = cv2.imread("Kode/patterns/" + path[-8:-1] + ".png") #get automatically based on stretch?

#invert, perhaps not necessary 
#pattern = cv2.bitwise_not(pattern, pattern) +1 617 6859628

#make grayscale
pattern_og = cv2.cvtColor(pattern_og, cv2.COLOR_BGR2GRAY)

#extract all contours
pat_contours, pat_hierarchy = cv2.findContours(pattern_og, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

pattern = pattern_og.copy()

#create mask 
pat_mask = np.zeros((pattern.shape),np.uint8)
cv2.drawContours(pat_mask,pat_contours,-1,255,-1)
#cv2.drawContours(pat_mask,pat_contours,-1,0,2)

x,y,w,h = cv2.boundingRect(pat_mask)
cv2.rectangle(pattern,(x,y),(x+w,y+h),(0,255,0),2)

#crop and resize
pattern = pattern[y:y+h, x:x+w]
dim = pattern.shape
dim = [int(height*stretch), height ]
pattern = cv2.resize(pattern, dim)
if show_steps:
    cv2.imshow("pattern - resized and cropped", pattern)
    cv2.waitKey()
    cv2.destroyAllWindows()
#------------------
#overlay
#extract all new contours (after crop and resize)
pat_cnt, pat_hierarchy = cv2.findContours(pattern, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

pat_mask = np.zeros((pattern.shape),np.uint8)
cv2.drawContours(pat_mask,pat_cnt,-1,255,-1)
cv2.drawContours(pat_mask,pat_cnt,-1,0,2)

x,y,w,h = cv2.boundingRect(pat_mask)
cv2.rectangle(pattern,(x,y),(x+w,y+h),(0,255,0),2)

overlay = image.copy()
cv2.drawContours(overlay,pat_cnt,-1,(175, 102, 61),-1)
cv2.drawContours(overlay,pat_cnt,-1,0,1)
if show_steps:
    cv2.imshow("overlay", overlay) 
    cv2.waitKey()
    cv2.destroyAllWindows()
cv2.imwrite(path + "overlay.png",overlay)

#intersections
file = open(path+'manual.txt', 'w')
file.write('threshold: ' + str(threshold) + '\n')
if threshold>0:
    file.write('flux direction: ' + flux)

file.write('\n\n')
blank = np.zeros((pattern.shape), np.uint8)
blank_og = np.zeros((pattern_og.shape), np.uint8)
mag = cv2.cvtColor(blank_og.copy(), cv2.COLOR_GRAY2BGR)
magupleft = []
magupright = [] 
magdownleft = [] 
magdownright = []
undecided = []
for i in range(len(pat_cnt)):
    try: 
        print(i)
        intensities = []
        line = blank.copy()
        pattern = cv2.drawContours(blank.copy(),[pat_cnt[i]],-1,255,-1)
        [vx, vy, x, y] = cv2.fitLine(pat_cnt[i], cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x*vy/vx)+y)
        righty = int(((line.shape[1]-x)*vy/vx)+y)
        cv2.line(line, (line.shape[1]-1, righty), (0, lefty), 255, 2)
        intersection = cv2.bitwise_and(pattern, line)
        #cv2.imshow("intersection", intersection)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        pts = np.where(intersection == 255)
        intensities = gray[pts[0], pts[1]]
        top = np.average(intensities[:int(len(intensities)/4)])
        bottom = np.average(intensities[int(len(intensities)*3/4):])
        if vy<0: #magnet points towards top right
            if top-bottom>threshold: #top is lighter than bottom
                magdownleft.append(pat_contours[i])
            elif bottom-top>threshold: #bottom is lighter than top
                magupright.append(pat_contours[i])
            else: 
                if flux == 'up': 
                    magupright.append(pat_contours[i])
                    file.write(str(i)  + ': upright\n')
                elif flux == 'down':
                    magdownleft.append(pat_contours[i])
                    file.write(str(i)  + ': downleft\n')
                elif flux == 'right':
                    magupright.append(pat_contours[i])
                    file.write(str(i) + ':uprightn\n')
                elif flux == 'left':
                    magdownleft.append(pat_contours[i])
                    file.write(str(i) + ':downleft\n')
        else: #magnet points towards top left
            if top-bottom>threshold: #top is lighter than bottom
                magdownright.append(pat_contours[i])
            elif bottom-top>threshold: #bottom is lighter than top
                magupleft.append(pat_contours[i])
            else:  
                if flux == 'up': 
                    magupleft.append(pat_contours[i])
                    file.write(str(i)  + ': upleft\n')
                elif flux == 'down':
                    magdownright.append(pat_contours[i])
                    file.write(str(i)  + ': downright\n')
                elif flux == 'right':
                    magdownright.append(pat_contours[i])
                    file.write(str(i) + ':downrightn\n')
                elif flux == 'left':
                    magupleft.append(pat_contours[i])
                    file.write(str(i) + ':upleft\n')
    except IndexError:
        print(i, 'no')

file.close()

mag = cv2.drawContours(mag, magupleft, -1, (55, 255, 0), -1)
magupleft_img = cv2.drawContours(blank_og.copy(), magupleft, -1, 255, -1)
mag = cv2.drawContours(mag, magupright, -1, (0, 172, 255), -1)
magupright_img = cv2.drawContours(blank_og.copy(), magupright, -1, 255, -1)
mag = cv2.drawContours(mag, magdownleft, -1, (255, 78, 0), -1)
magdownleft_img = cv2.drawContours(blank_og.copy(), magdownleft, -1, 255, -1)
mag = cv2.drawContours(mag, magdownright, -1,(176, 0, 255) , -1)
magdownright_img = cv2.drawContours(blank_og.copy(), magdownright, -1, 255, -1)
mag = cv2.drawContours(mag, undecided, -1, (255, 255, 255), -1)
if show_steps:
    cv2.imshow("magnetization", mag)
    cv2.waitKey()
    cv2.destroyAllWindows()
cv2.imwrite(path + "magnetization.png", mag)
cv2.imwrite(path + "magupleft.png", magupleft_img)
cv2.imwrite(path + "magupright.png", magupright_img)
cv2.imwrite(path + "magdownleft.png", magdownleft_img)
cv2.imwrite(path + "magdownright.png", magdownright_img)

PEEM = np.full((pattern_og.shape), 128, np.uint8)
PEEM = cv2.drawContours(PEEM, magupleft, -1, (255, 255, 255), -1)
PEEM = cv2.drawContours(PEEM, magdownleft, -1, (255, 255, 255), -1)
PEEM = cv2.drawContours(PEEM, magupright, -1, (0, 0, 0), -1)
PEEM = cv2.drawContours(PEEM, magdownright, -1, (0, 0, 0), -1)
PEEM = cv2.GaussianBlur(PEEM, (47, 47), 0)
cv2.imwrite(path + "PEEM.png", PEEM)