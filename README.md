# Image-Classification
Three classifiers that decide the correct orientation of a given image.

# Motivation
These days, all modern digital cameras include a sensor that detects which way the camera is being held when a photo is taken. This metadata is then included in the image file, so that image organization programs know the correct orientation i.e., which way is up, in the image. But for photos scanned in from film or from older digital cameras, rotating images to be in the correct orientation must typically be done by hand.

# Data
A dataset of images from the Flickr photo sharing website. The images were taken and uploaded by real users from across the world. Each image is rescaled to a very tiny micro-thumbnail of 8*8 pixels, resulting in an 8*8*3 = 192 dimensional feature vector.
The text files have one row per image, where each row is formatted like:
photo_id correct_orientation r11 g11 b11 r12 g12 b12 ...
where:
• photo id is a photo ID for the image.
• correct orientation is 0, 90, 180, or 270. Note that some small percentage of these labels may be wrong because of noise in the data; this is just a fact of life when dealing with data from real-world sources.
• r11 refers to the red pixel value at row 1 column 1, r12 refers to red pixel at row 1 column 2, etc., each in the range 0-255.

train.txt - 40000 images
test.txt - 1000 images

# Algorithms implemented
K-Nearest Neighbor
Adaboost
Neural Networks

Use the following command to run the code:
python orient.py train_file.txt test_file.txt <mode>

# Team Members
[Manan Papdiwala]
[Anurag Jain]

# CourseWork
CSCI-B551: [Elements of Artificial Intelligence] (https://www.soic.indiana.edu/graduate/courses/index.html?number=b551&department=CSCI) by Professor [David Crandall] (http://www.soic.indiana.edu/all-people/profile.html?profile_id=183)

# Book Referenced
Artificial Intelligence: A Modern Approach (3rd Ed.) by Stuart Russell and Peter Norvig
