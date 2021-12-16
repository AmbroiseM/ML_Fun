# Image segmentation with level set

![alt text-1](https://github.com/AmbroiseM/ML_Fun/blob/main/old-projects/Contour%20actif/resultats/avion2_seg.png)             ![alt-text-2](https://github.com/AmbroiseM/ML_Fun/blob/main/old-projects/Contour%20actif/resultats/photograph_seg.png)

This is an old project of mine on image segmentation based on two level set methods:

* Here, [Sparse field method](https://github.com/AmbroiseM/ML_Fun/blob/main/old-projects/Contour%20actif/Lankton%20-%202009%20-%20SFM%20Tech%20Report.pdf), from Withaker, explained nicely by Lankton
* [Fast two cycle](https://github.com/AmbroiseM/ML_Fun/blob/main/old-projects/Contour%20actif/Shi%20-%20TIP%20-%202008%20-%20FTC.pdf), from Shi and Karl
  
SFM's code are: [sfm_start.py](https://github.com/AmbroiseM/ML_Fun/blob/main/old-projects/Contour%20actif/sfm_start.py), [calcul_sfm.py](https://github.com/AmbroiseM/ML_Fun/blob/main/old-projects/Contour%20actif/calcul_sfm.py).

FTC's code are: [debut_ftc.py](https://github.com/AmbroiseM/ML_Fun/blob/main/old-projects/Contour%20actif/debut_ftc.py), [calcul_ftc.py](https://github.com/AmbroiseM/ML_Fun/blob/main/old-projects/Contour%20actif/calcul_ftc.py).

## How to run 
To have your segmented image, run [display_results.py](https://github.com/AmbroiseM/ML_Fun/blob/main/old-projects/Contour%20actif/display_results.py) where you'll need to initialize a mask on the part of the image you want to segmentate .

I say it was an old project so i didn't had enough time to make a cleaner code, sorry :grimacing: . 

Hope it can help you, whoever you are if you came across this repo :smiley: .
