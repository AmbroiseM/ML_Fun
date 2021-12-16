import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2

def load_images_from_folder(folder):
    images = []
    print("first .......")
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = np.array(img)
            images.append(img)
    return images

def load_images_from_folder_bis(liste):
    images = []
    print("bis .......")
    for filename in liste:
        img = cv2.imread(filename)
        if img is not None:
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = np.array(img)
            images.append(img)
    return images


# =============================================================================
#                           Charge les images de masks   
# =============================================================================
# path_mask = r"C:\Users\david\Desktop\M2\MLA\Projet\covid-chestxray-dataset-master\annotations\lungVAE-masks"
# img_mask = load_images_from_folder(path_mask)
# path = r"C:\Users\david\Desktop\M2\MLA\Projet\covid-chestxray-dataset-master"
# os.chdir(path)
# np.save("image_mask.npy",img_mask,allow_pickle=True)

# =============================================================================
#                           Recupere les noms des fichiers d'interets
# =============================================================================
# filenames = []

# for filename in os.listdir(path_mask):
#     filename = filename.replace("_mask","")
#     filenames.append(filename.replace(".png",""))

# =============================================================================
#                           Recupere les images d'interets
# =============================================================================
# path_images = r"C:\Users\david\Desktop\M2\MLA\Projet\covid-chestxray-dataset-master\images/"
# os.chdir(path_images)
# x = []
# images = []
# for i in filenames:
#     for name in os.listdir(path_images):
#         z = name 
#         name = os.path.splitext(name)[0]
#         if name == i:
#             x.append(name)
#             img = cv2.imread(z)
#             img = np.array(img)
#             images.append(img)

# root_path = r"C:\Users\david\Desktop\M2\MLA\Projet\covid-chestxray-dataset-master"
# os.chdir(root_path)
# np.save("image.npy",images,allow_pickle=True)
# =============================================================================
#                          MASK + IMG = IMAGE SEGMENTER 
# =============================================================================
# res = []
# # np.save("image_mask.npy",img_mask,allow_pickle=True)
# for i in range (len(images)):
#     res.append(cv2.bitwise_and(images[i],img_mask[i]))

# np.save("Segmented.npy",res,allow_pickle=True)

Segmented = np.load("Segmented.npy",allow_pickle=True)

w=10
h=10
fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 2
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(Segmented[i])
plt.show()
