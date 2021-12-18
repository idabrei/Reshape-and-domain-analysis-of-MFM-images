import cv2
import numpy as np
import matplotlib.pyplot as plt

show_steps = False
show_vertice = False
path = "Figures/Bilder/resultater/as-grown/p340_12/"
stretch = int(path[-3:-1])/10
print(stretch)

mag = cv2.imread(path + "magnetization.png")

if show_steps:
    cv2.imshow("mag", mag)
    cv2.waitKey()
    cv2.destroyAllWindows()

gray = cv2.cvtColor(mag, cv2.COLOR_BGR2GRAY)

cnt, h = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = np.zeros((mag.shape),np.uint8)
cv2.drawContours(contours,cnt,-1,255,-1)

domains = np.zeros((mag.shape), np.uint8)
arrows = np.zeros((mag.shape), np.uint8)
#domains = mag.copy()
mag_00 = 0
mag_01 = 0
mag_11 = 0
mag_10 = 0
mag_1n1 = 0
mag_0n1 = 0
mag_n1n1 = 0
mag_n10 = 0
mag_n11 = 0
test_domains = domains.copy()
tot_dir = [0, 0]

for i in range(len(cnt)):  
    try:
        [vx, vy, x, y] = cv2.fitLine(cnt[i], cv2.DIST_L2, 0, 0.01, 0.01)
        if vy> 0:
            dir = [0, 0]
            mask_a = cv2.drawContours(np.zeros((mag.shape[:2]), np.uint8), [cnt[i]], -1, 255, -1)
            mask_b = cv2.drawContours(np.zeros((mag.shape[:2]), np.uint8), [cnt[i + 25]], -1, 255, -1)
            mask_c = cv2.drawContours(np.zeros((mag.shape[:2]), np.uint8), [cnt[i + 26]], -1, 255,  -1)
            mask_d = cv2.drawContours(np.zeros((mag.shape[:2]), np.uint8), [cnt[i + 51]], -1, 255, -1)
            if show_vertice:
                vertex = mask_a + mask_b + mask_c + mask_d
                cv2.imshow("vertex", vertex)
                cv2.waitKey()
                cv2.destroyAllWindows()
            a = cv2.mean(mag, mask_a)
            b = cv2.mean(mag, mask_b)
            c = cv2.mean(mag, mask_c)
            d = cv2.mean(mag, mask_d)
            if abs(a[0]-d[0])<10 and abs(a[1]-d[1])<10 and abs(a[2]-d[2])<10:
                if a[1] > 250 and a[2] > 250:
                    print('undecided')
                elif a[1] > 250:
                    dir[0] -= 0.5
                    dir[1] += 0.5
                else: 
                    dir[0] += 0.5
                    dir[1] -= 0.5
            if abs(c[0]-b[0])<10 and abs(b[1]-c[1])<10 and abs(c[2]-b[2])<10:
                if b[1] > 250 and b[2] > 250:
                    print('undecided')
                elif b[0] > 250:
                    dir[0] -= 0.5
                    dir[1] -= 0.5
                else: 
                    dir[0] += 0.5
                    dir[1] += 0.5
            x,y,w,h = cv2.boundingRect(mask_a)
            if dir[0] == dir[1]:
                if dir[0] == 0: 
                    mag_00 += 1
                    cv2.rectangle(domains,(x, y-h),(x+w,y),(0, 0, 0),-1)
                if dir[0] > 0:
                    mag_11 += 1
                    cv2.rectangle(domains,(x, y-h),(x+w,y),(0, 170, 255),-1)
                    cv2.arrowedLine(arrows, (x, y), (x+w, y-h), (0, 170, 255), 1)
                    tot_dir[0] += 0.5
                    tot_dir[1] += 0.5
                if dir[0] < 0:
                    mag_n1n1 += 1
                    cv2.rectangle(domains,(x, y-h),(x+w,y),(255, 0, 0),-1)
                    cv2.arrowedLine(arrows, (x+w, y-h), (x, y), (255, 0, 0), 1)
                    tot_dir[0] -= 0.5
                    tot_dir[1] -= 0.5
            elif dir[0] == 0 and dir[1] > 0:
                mag_01 += 1
                cv2.rectangle(domains,(x, y-h),(x+w,y),(0, 255, 255),-1)
                cv2.arrowedLine(arrows, (x+int(w/2), y), (x+int(w/2), y-h), (0, 255, 255), 1)
                tot_dir[1] += 1
            elif dir[0] == 0 and dir[1] < 0:
                mag_0n1 += 1
                cv2.rectangle(domains,(x, y-h),(x+w,y),(255, 0, 170),-1)
                cv2.arrowedLine(arrows, (x+int(w/2), y-h), (x+int(w/2), y), (255, 0, 170), 1)
                tot_dir[1] -= 1
            elif dir[1] == 0 and dir[0] > 0:
                mag_10 += 1
                cv2.rectangle(domains,(x, y-h),(x+w,y),(0, 0, 255),-1)
                cv2.arrowedLine(arrows, (x, y-int(h/2)), (x+w, y - int(h/2)), (0, 0, 255), 1)
                tot_dir[0] += 1
            elif dir[1] == 0 and dir[0] < 0:
                mag_n10 += 1
                cv2.rectangle(domains,(x, y-h),(x+w,y),(255, 255, 0),-1) 
                cv2.arrowedLine(arrows, (x+w, y-int(h/2)), (x, y - int(h/2)), (255, 255, 0), 1)
                tot_dir[0] -= 1
            elif dir[0] < 0 and dir[1] > 0:
                mag_n11 += 1
                cv2.rectangle(domains,(x, y-h),(x+w,y),(0, 255, 0),-1)
                cv2.arrowedLine(arrows, (x+w, y), (x, y-h), (0, 255, 0), 1)
                tot_dir[0] -= 0.5
                tot_dir[1] += 0.5
            elif dir[0] > 0 and dir[1] < 0:
                mag_1n1 += 1
                cv2.rectangle(domains,(x, y-h),(x+w,y),(170, 0, 255),-1)
                cv2.arrowedLine(arrows, (x, y-h), (x+w, y), (170, 0, 255), 1)
                tot_dir[1] += 0.5
                tot_dir[1] -= 0.5
        else: 
            mask_a = cv2.drawContours(np.zeros((mag.shape[:2]), np.uint8), [cnt[i]], -1, 255, -1)
            x,y,w,h = cv2.boundingRect(mask_a)
            cv2.rectangle(test_domains, (x, y-h),(x+w,y),(255, 255, 255),-1)
    except IndexError: 
        break

cv2.imwrite(path + "test_domains.png", test_domains)
if show_steps:
    cv2.imshow("domains", domains)
    cv2.waitKey()
    cv2.destroyAllWindows()
if show_steps:
    cv2.imshow("arrows", arrows)
    cv2.waitKey()
    cv2.destroyAllWindows()

print(tot_dir)

cv2.imwrite(path + "domains.png", domains)
cv2.imwrite(path + "arrows.png", arrows)

with open(path[:-8] + "stats.csv", 'a') as file:
    string = path[-8:-1] + ',' + str(mag_00) + ',' + str(mag_01) + ',' + str(mag_11) + ',' + str(mag_10) + ',' + str(mag_1n1) + ',' + str(mag_0n1) + ',' + str(mag_n1n1) + ',' + str(mag_n10) + ',' + str(mag_n11) + ',' + str(tot_dir) + '\n'
    file.write(string) 