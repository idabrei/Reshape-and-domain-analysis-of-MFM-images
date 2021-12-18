import cv2
import numpy as np
import matplotlib.pyplot as plt

show_steps = False
path = "Figures/Bilder/resultater/test/p340_10/"

sim_MFM = cv2.imread(path + "sim-MFM.png")
resize = cv2.imread(path + "resize.png")
#pattern = cv2.imread(path + "domains.png")
pattern = cv2.imread("kode/patterns/" + path[-8:-1] + ".png")
uncertainty = np.zeros((pattern.shape[:2]), np.uint8)

mags, h = cv2.findContours(cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

pattern = cv2.resize(pattern, sim_MFM.shape[:-1])

def SSD(difference): 
    difference = difference/255
    difference = np.square(difference)
    SSD = np.sum(difference)
    SSD_avg = SSD/len(difference.ravel())
    return SSD_avg

resize = cv2.GaussianBlur(resize, (37,37), 0)

#crop   
thresh_dark = cv2.adaptiveThreshold(cv2.cvtColor(sim_MFM, cv2.cv2.COLOR_BGR2GRAY), 255, 1, 1,161, 7 )
thresh_light = cv2.adaptiveThreshold(cv2.bitwise_not(cv2.cvtColor(sim_MFM,cv2.COLOR_BGR2GRAY)), 255, 1, 1,161, 7 )
if show_steps:
    cv2.imshow("thresh", thresh_dark)
    cv2.waitKey() 
    cv2.destroyAllWindows()
    cv2.imshow("thresh", thresh_light)
    cv2.waitKey() 
    cv2.destroyAllWindows()
thresh = cv2.add(thresh_dark, thresh_light)
rect = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
x,y,w,h = cv2.boundingRect(thresh)
cv2.rectangle(rect,(x, y),(x+w,y+h),(55,98,226),30)
if show_steps:
    cv2.imshow("rectangle", rect)
    cv2.waitKey()
    cv2.destroyAllWindows()

val =np.zeros((10,10))
start_x = -30
start_y = -20
x_range = 10
y_range = 10
minima_found = [False, False]
while not minima_found[0] or not minima_found[1]:
    for i in range(x_range):
        frame_x = start_x+i
        for j in range(y_range):
            frame_y = start_y +j
            fit = sim_MFM[y- frame_y:y+h+frame_y, x-frame_x:x+w+frame_x]
            fit = cv2.resize(fit,(len(resize[0]), len(resize)))
            difference = cv2.absdiff(fit, resize)
            val[i][j] = SSD(difference)
    frame = np.where(val == np.min(val))
    print(start_x + frame[0], start_y + frame[1])
    try:
        frame_x = int(start_x + frame[0])
        if frame_x == start_x and not minima_found[0]:
            print('WARNING: frame_x is at the edge of tested values. Changing test-interval and running again.  ')
            start_x -= 7
        elif frame_x == start_x+9 and not minima_found[0]:
            print('WARNING: frame_x is at the edge of tested values. Changing test-interval and running again. ')
            start_x += 7
        else: 
            start_x = frame_x
            x_range = 1
            minima_found[0] = True
    except TypeError:
        start_x += 8
    try:
        frame_y = int(start_y + frame[1])
        if frame_y == start_y and not minima_found[1]:
            print('WARNING: frame_y is at the edge of tested values. Changing test-interval and running again. ')
            start_y -= 7
        elif frame_y == start_y + 9 and not minima_found[1]:
            print('WARNING: frame_y is at the edge of tested values. Changing test-interval and running again. ')
            start_y += 7
        else: 
            start_y = frame_y
            y_range = 1
            minima_found[1] = True
    except TypeError:
        start_y += 8

print('minima found')
#frame_x = -60
#frame_y = -60
sim_MFM = sim_MFM[y-frame_y:y+h+frame_y, x-frame_x:x+w+frame_x]
sim_MFM = cv2.resize(sim_MFM,(len(resize[0]), len(resize)))
sim_MFM = cv2.cvtColor(sim_MFM, cv2.COLOR_BGR2GRAY)
resize = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
pattern = pattern[y-frame_y:y+h+frame_y, x-frame_x:x+w+frame_x]
pattern = cv2.resize(pattern,(len(resize[0]), len(resize)))
pattern= cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

pat_cnt, hierarchy = cv2.findContours(pattern, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#brightness and contrast change

x = np.linalg.solve(np.array([[np.min(sim_MFM), 1], [np.max(sim_MFM), 1]]), np.array([np.min(resize)+50, np.max(resize)-50]))

for i in range(len(sim_MFM)):
    for j in range(len(sim_MFM[0])):
        sim_MFM[i][j] = int(sim_MFM[i][j]*x[0] + x[1])

difference =cv2.absdiff(sim_MFM,resize)
if show_steps:
    cv2.imshow("sim_MFM", sim_MFM)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imshow("resize", resize) 
    cv2.waitKey()
    cv2.destroyAllWindows()
cv2.imwrite(path + "difference.png", difference)
simexpdiff = cv2.hconcat([resize, sim_MFM, difference])

cv2.imwrite(path + "simexpdiff.png", simexpdiff)

flat = difference.flatten()
plt.hist(flat, bins = 10)
plt.savefig(path + "pix-histogram.png")

overlay_diff = cv2.drawContours(difference, pat_cnt, -1, 255, 1)
cv2.imwrite(path + 'overlay_diff.png', overlay_diff)
intensities = []
for i in range(len(pat_cnt)): 
    mask = cv2.drawContours(np.zeros((difference.shape[:2]), np.uint8), [pat_cnt[i]], -1, 255, -1)
    ints = cv2.mean(difference, mask)
    intensities.append(ints[0])
    uncertainty = cv2.drawContours(uncertainty, [mags[i]], -1, round(ints[0]), -1)

plt.hist(intensities, bins = 20)
plt.savefig(path + "histogram.png")

cv2.imwrite(path + "uncertainties.png", uncertainty)

print(SSD(difference))

#cv2.putText(difference, 'mismatch: {:.2f}'.format(mismatch), (int(len(difference[0])/15), int(len(difference)/15)), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 3, color = (255, 255, 255), thickness = 3)

if show_steps:
    cv2.imshow("difference", difference)
    cv2.waitKey()
    cv2.destroyAllWindows() 
    cv2.imshow("", simexpdiff)
    cv2.waitKey()
    cv2.destroyAllWindows

with open(path[:-8] + "uncertainty.csv", 'a') as file:
    string = path[-8:-1] + ',' + str(SSD(difference)) + '\n'
    file.write(string) 

