import os
import cv2


path = r"C:\Users\West\Downloads\lfw_funneled"
newrootpath = r'2'
newpathtrain = newrootpath+r'\\train'
newpathtest = newrootpath+r'\\test'
os.mkdir(newrootpath)
os.mkdir(newpathtrain)
os.mkdir(newpathtest)
count = 0
d = 0.2

for folder in os.listdir(path):

    imgfolders = os.listdir(path+'\\'+folder)
    len1 = len(imgfolders)
    if len1 < 9:
        continue

    valid_photos = []
    for file in imgfolders:
        count += 1
        print(count)
        original_image = cv2.imread(path+'\\'+folder+'\\'+file)

        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        detected_faces = face_cascade.detectMultiScale(grayscale_image)
        if detected_faces == () or len(detected_faces.tolist()) > 1:
            continue
        (column, row, width, height) = detected_faces[0]
        dx = int(d*width)
        dy = int(d*height)

        original_image = original_image[row-dy:row +
                                        height+dy, column-dx:column+width+dx]
        if 0 in original_image.shape:
            continue
        # отбираем нормальные фотографии
        valid_photos.append((file, original_image))
    len_valid = len(valid_photos)
    if len_valid < 7:
        continue

    os.mkdir(newpathtrain+'\\'+folder)
    os.mkdir(newpathtest+'\\'+folder)
    for n, (file, original_image) in enumerate(valid_photos):
        if n == len_valid-1:
            # добавляем в test последнюю фотку

            cv2.imwrite(newpathtest+'\\'+folder+'\\'+file, original_image)
        else:

            cv2.imwrite(newpathtrain+'\\'+folder+'\\'+file, original_image)
