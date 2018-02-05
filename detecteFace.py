import cv2
import numpy as np
import math
import scipy.stats
import  matplotlib.pyplot as plt

image_con = ["image1.jpg","image2.jpg","image3.jpg","image4.jpg"]
image_test = ["image1.jpg","image2.jpg","image3.jpg","image4.jpg"]
image_enter = "med1.jpg"
intersection = []
intersection1 = []
image_test1 = []
caree = []
caree1 = []
image_test2 = []
def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

def carre(hist_1, hist_2):
    chi2 = cv2.compareHist(hist_1, hist_2, 1)
    return int(chi2)


def processing(image_tes , image_ent , intersect , careeee):
    img3 = cv2.imread(image_tes, 1)
    img4 = cv2.imread(image_tes, 0)
    img = cv2.imread(image_ent, 1)
    img1 = cv2.imread(image_ent, 0)
    home1 = cv2.resize(img1, (600, 600))
    home2 = cv2.resize(img4, (600, 600))

    home3 = cv2.resize(img, (600, 600))
    home4 = cv2.resize(img3, (600, 600))

    resized_image1 = cv2.resize(img, (600, 600))
    resized_image = cv2.resize(img, (600, 600))
    imgh = cv2.resize(img1, (600, 600))
    image1 = cv2.resize(img1, (600, 600))
    #imgARGB = cv2.cvtColor(resized_image,cv2.COLOR_RGB2RGBA)
    imghsv = cv2.cvtColor(resized_image,cv2.COLOR_RGB2HSV)
    imgycrcb = cv2.cvtColor(resized_image,cv2.COLOR_RGB2YCR_CB)
    imghsv_s = cv2.cvtColor(home4, cv2.COLOR_RGB2HSV)
    imgh, imgs,imgv = cv2.split(imghsv)
    imgh_s, imgs_s,imgv_s = cv2.split(imghsv_s)
    RGB_local = []
    RGB_local_x = []
    RGB_local_y = []

    print(imghsv[0, 0] )
    for x in range(600):
        for y in range(600):
            RGB=resized_image[y, x]
            HSV = imghsv[y, x]
            YCRCB = imgycrcb[y, x]
            if ((80 <= HSV[2] <= 145 and 90 <= HSV[1] <= 155) or (30 <= YCRCB[0] <= 155 and 90 <= YCRCB[1] <= 120 and 110 <= YCRCB[2] <= 150) or (RGB[2] > 95 and RGB[1] > 40 and RGB[0] > 20 and RGB[2] > RGB[1] and RGB[2] > RGB[0] and abs(RGB[2] - RGB[1]) > 15)):
                    resized_image[y, x] = RGB
                    RGB_local.append([RGB[0],RGB[1],RGB[2]])
                    RGB_local_x.append(x)
                    RGB_local_y.append(y)
            else:
                resized_image[y, x] = [0, 0, 0]
    resized_image2 = cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)
    for x in range(1,600-1):
        for y in range(1,600-1):
            image1[x, y] = 0
            image1[x,y] = math.sqrt((abs(resized_image2[x, y] - resized_image2[x+1,y-1] ) + abs(resized_image2[x, y] - resized_image2[x+1,y-1] ))**2)


    hist_1 = cv2.calcHist([imgs], [0], None, [256], [0, 256])
    hist_2 = cv2.calcHist([imgs_s], [0], None, [256], [0, 256])


    intersect.append(return_intersection(hist_1,hist_2))
    careeee.append(carre(hist_1, hist_2))
    xx = 0
    xxx = 600
    yy = 0
    yyy = 600
    #for x in range(800):
    for y in range(len(RGB_local_x)):
        if RGB_local_y[y] > yy  :
            yy = RGB_local_y[y]
        if RGB_local_y[y] < yyy  :
            yyy = RGB_local_y[y]
        if RGB_local_x[y] > xx  :
            xx = RGB_local_x[y]
        if RGB_local_x[y] < xxx  :
            xxx = RGB_local_x[y]

    #print ("xx = ",xx," xxx = ",xxx , " yy = ", yy , " yyy = ",yyy)

    cv2.rectangle(resized_image1,(xx,yy),(xxx,yyy),(0,0,255),2)

    cv2.imshow('image Normal', resized_image1)
    #cv2.imwrite('image'+str(k)+'.jpg', resized_image)
    cv2.imshow('image skim', resized_image)
    #cv2.imshow('image 2', home4)







def main():
    for img in image_test:
        processing(img,image_enter,intersection,caree)

    print(intersection)
    for x in range(len(intersection)):
        intersection1.append(max(intersection))
        i = 0
        for y in intersection:
            if (max(intersection) == y):
                image_test1.append(image_test[i])
                intersection[i] = -1
                break
            i+=1
    print(intersection1)
    print(image_test)
    print(image_test1)
    i = 0
    for x in image_test1:
        imag  = cv2.resize(cv2.imread(x), (400, 400))
        image = "image"+str(i)
        cv2.imshow(image, imag)
        i += 1
    print(caree)
    for x in range(len(caree)):
        caree1.append(min(caree))
        i = 0
        for y in caree:
            if (min(caree) == y):
                image_test2.append(image_test[i])
                caree[i] = 9999999999999
                break
            i += 1
    print(caree1)
    print(image_test)
    print(image_test2)
    i = 0
    for x in image_test2:
        imag = cv2.resize(cv2.imread(x), (400, 400))
        image = "imagee" + str(i)
        cv2.imshow(image, imag)
        i += 1
main()

cv2.waitKey(0)
cv2.destroyAllWindows()


