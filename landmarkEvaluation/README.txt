sdmEvaluation
=============

Evaluate an SDM landmark model on a set of images and generate result images plus a results.txt with errors normalized by the inter-eye distance (IED) of every image.
The V&J face-detector from OpenCV is run on the images and the landmark model initialized with this bounding-box. If a face could not be found, the image is skipped.

Landmark lists from Zhenhua on iBug-LFPW:
"Full"-list: 9 18 20 22 23 25 27 28 31 32 34 36 37 40 43 46 49 52 55 58 63 67 
me17-list:     18    22 23    27       32 34 36 37 40 43 46 49 52 55 58

sdmEvaluation -v -i C:\Users\Patrik\Documents\GitHub\data\iBug_lfpw\testset -l C:\Users\Patrik\Documents\GitHub\data\iBug_lfpw\testset -t ibug -f "C:\opencv\opencv_2.4.8_prebuilt\sources\data\haarcascades\haarcascade_frontalface_alt2.xml" -m C:\Users\Patrik\Documents\GitHub\sdm_lfpw_tr100_20lm_10s_5c_r0.5_1_0_opencvsift.txt -o C:\Users\Patrik\Documents\GitHub\sdmEvalOut\ --landmarks-to-compare 18 22 23 27 32 34 36 37 40 43 46 49 52 55 58
