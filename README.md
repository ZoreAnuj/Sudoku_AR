# AR Sudoku Solver
This is a basic augmented reality project I picked up to learn more about computer vision and gain experience using the OpenCV library in Python. Upon being shown an incomplete sudoku board on a camera through a live video feed, it augments the missing digits back onto the board in real-time.]

# Input / Output
<p align="middle">
  <img src="1_input_image.jpg" height=320/>
  <img src="7_final_image.jpg" height=320/>
</p>

# Approach Overview
To solve this problem, I first isolate the sudoku board from the rest of the environment. I then evenly split up the board to crop even smaller images of the 81 cells. Using a convolutional neural net trained for 10 potential image inputs (digits from 1-9 as well as a blank input), I was able to determine exactly what digits are present and where they lie on the board. I reformatted this information to fit the sudoku solver and upon parsing the solved return value, I populated the cells with missing digits to produce the final result.

# Image Processing
This project was quite heavy on image processing as I needed to ensure I got the most accurate results for digit localization and recognition. In this subsection, I will briefly go over some of the transformations I applied and explain why I believed they were necessary.

**1. Input Image**  
<img src="1_input_image.jpg" height=300/>  
I first converted this input image to grayscale because I intend on applying adaptive thresholding, which is an image transformation which is generally applied to grayscale images.

**2. Grayscale Image**  
<img src="2_input_image_gray.jpg" height=300/>  
I want to apply adaptive thresholding to this image to convert it to a binary image, where each pixel is either white or black. This will effectively distinguish the digits and the sudoku board's grid from the whitespace while also ridding the image of small specks of noise.

**3. Adaptive Thresholding**  
<img src="3_apply_adaptive_thresholding.jpg" height=300/>  
Evidently, the contours of this image are emphasized much more than the original, especially for the digits.  
<img src="visual_media/4_find_sudoku_board.jpg" height=300/>  
With these better contours, I can easily find and isolate the sudoku board by searching within image for the contour with the largest area.

**4. Warp Perspective**  
<img src="5_warp_perspective.jpg" height=300/>  
I applied a perspective transform here to get a cleanly cropped image of the sudoku board. I could have directly cropped it as well; however, with a perspective transform, I can also correct the cropped image to appear completely flat even if the original image has uneven depth across the board.

**5. Crop Digit**  
<img src="original_digit.jpg" height=300/>  
It is now easy for me to divide the previous image into 9 rows and 9 columns to process each cell individually. This is an example of one such cell. There is still a lot of noise within the image accompanying the digit 1. To minimize the risk of misclassification, I apply cleansing techniques such as flood filling, erosion and dilation to remove these islands of noise in the center and on the edges of the image. I also try to recenter the image so that most of the data lies in the middle, as that is the type of input data the classifier was trained with.

**6. Clean Digit**  
<img src="cleaned_digit.jpg" height=300/>  
With a much cleaner image, the probability of correct classification is much higher.

**7. Populate Empty Cells**  
<img src="6_resize_to_fit_on_input_image.jpg" height=300/>  
Most of the difficult work is now complete. I am simply populating empty cells with the required digits here.

**8. Augment New Board**  
<img src="7_final_image.jpg" height=300/>  
As I know the coordinates of the rectangle we cropped earlier, I just replace that subsection of the original image with this new board.
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Mar  3 01:08:02 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Mar  4 00:59:39 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Mar  5 01:12:54 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Mar  6 00:37:15 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Mar  7 01:41:42 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Mar  8 01:24:38 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Mar  9 01:22:39 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Mar 10 00:56:50 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Mar 11 01:00:14 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Mar 12 01:12:46 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Mar 13 00:37:46 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Mar 14 01:40:33 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Mar 15 01:51:39 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Mar 16 01:49:37 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Mar 17 01:09:50 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Mar 18 01:00:21 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Mar 19 01:14:37 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Mar 20 00:37:14 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Mar 21 01:43:41 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Mar 22 01:52:55 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Mar 23 01:50:50 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Mar 24 01:10:19 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Mar 25 01:01:42 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Mar 26 01:14:57 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Mar 27 00:38:09 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Mar 28 01:43:40 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Mar 29 01:54:46 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Mar 30 01:52:59 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Mar 31 01:12:25 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Apr  1 01:09:17 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Apr  2 01:16:11 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Apr  3 00:38:11 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Apr  4 01:44:09 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Apr  5 01:54:55 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Apr  6 01:51:51 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Apr  7 01:11:42 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Apr  8 01:01:55 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Apr  9 01:15:46 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Apr 10 00:38:26 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Apr 11 01:45:48 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Apr 12 01:55:20 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Apr 13 03:13:34 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Apr 14 01:12:47 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Apr 15 01:03:55 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Apr 16 01:17:46 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Apr 17 00:39:06 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Apr 18 01:43:51 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Apr 19 01:54:12 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Apr 20 01:54:46 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Apr 21 01:14:01 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Apr 22 01:03:05 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Apr 23 01:17:28 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Apr 24 00:39:43 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Apr 25 01:48:34 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Apr 26 01:56:40 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Apr 27 01:55:20 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Apr 28 01:13:30 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Apr 29 01:03:44 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Apr 30 01:18:22 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu May  1 00:45:06 UTC 2025
