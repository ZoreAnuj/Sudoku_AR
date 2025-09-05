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
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri May  2 01:49:20 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat May  3 01:59:29 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun May  4 02:00:37 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon May  5 01:15:02 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue May  6 01:04:59 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed May  7 01:19:43 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu May  8 00:41:13 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri May  9 01:50:27 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat May 10 01:58:39 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun May 11 01:58:29 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon May 12 01:15:42 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue May 13 01:06:12 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed May 14 01:19:32 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu May 15 00:40:03 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri May 16 01:52:33 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat May 17 02:01:47 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun May 18 02:00:09 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon May 19 01:17:05 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue May 20 01:07:00 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed May 21 01:21:15 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu May 22 00:40:51 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri May 23 01:51:40 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat May 24 02:00:27 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun May 25 02:02:50 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon May 26 01:15:42 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue May 27 01:05:27 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed May 28 01:21:22 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu May 29 00:41:24 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri May 30 01:50:33 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat May 31 02:03:02 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Jun  1 02:18:55 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Jun  2 01:17:35 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Jun  3 01:08:04 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Jun  4 01:22:16 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Jun  5 00:41:54 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Jun  6 01:52:51 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Jun  7 02:05:12 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Jun  8 02:05:12 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Jun  9 01:18:58 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Jun 10 01:08:52 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Jun 11 01:23:03 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Jun 12 00:41:50 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Jun 13 01:55:02 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Jun 14 02:03:59 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Jun 15 02:07:09 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Jun 16 01:18:09 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Jun 17 01:08:02 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Jun 18 01:22:16 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Jun 19 00:42:34 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Jun 20 01:54:52 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Jun 21 02:05:32 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Jun 22 02:06:29 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Jun 23 01:20:19 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Jun 24 01:08:27 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Jun 25 01:23:46 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Jun 26 00:42:33 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Jun 27 01:56:30 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Jun 28 02:05:23 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Jun 29 02:08:53 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Jun 30 01:20:43 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Jul  1 01:16:14 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Jul  2 01:23:20 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Jul  3 00:42:38 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Jul  4 01:56:00 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Jul  5 02:05:16 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Jul  6 02:07:18 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Jul  7 01:20:55 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Jul  8 01:09:20 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Jul  9 01:24:52 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Jul 10 00:43:08 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Jul 11 02:00:53 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Jul 12 02:21:24 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Jul 13 02:18:15 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Jul 14 01:22:07 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Jul 15 01:14:00 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Jul 16 01:26:50 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Jul 17 00:44:50 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Jul 18 02:04:04 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Jul 19 02:17:34 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Jul 20 02:20:13 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Jul 21 01:24:34 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Jul 22 01:12:26 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Jul 23 01:37:55 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Jul 24 00:45:01 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Jul 25 02:03:11 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Jul 26 02:19:12 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Jul 27 02:21:07 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Jul 28 01:25:49 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Jul 29 01:21:05 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Jul 30 01:39:18 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Jul 31 00:45:45 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Aug  1 02:22:23 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Aug  2 02:20:02 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Aug  3 02:23:17 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Aug  4 01:27:38 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Aug  5 01:15:49 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Aug  6 01:40:08 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Aug  7 00:47:04 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Aug  8 02:06:40 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Aug  9 02:09:18 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Aug 10 02:19:03 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Aug 11 01:23:12 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Aug 12 01:07:15 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Aug 13 01:23:37 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Aug 14 00:42:46 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Aug 15 01:57:49 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Aug 16 02:05:39 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Aug 17 02:05:55 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Aug 18 01:21:37 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Aug 19 01:06:05 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Aug 20 01:18:34 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Aug 21 00:38:46 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Aug 22 01:48:59 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Aug 23 01:57:28 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Aug 24 02:01:44 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Aug 25 01:14:33 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Aug 26 01:05:08 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Aug 27 01:17:01 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Aug 28 00:38:57 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Aug 29 01:45:39 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sat Aug 30 01:52:39 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Sun Aug 31 01:53:05 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Mon Sep  1 01:21:13 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Tue Sep  2 01:03:29 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Wed Sep  3 01:13:05 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Thu Sep  4 00:37:34 UTC 2025
Downloaded file from https://preview.redd.it/uxilb61s0fo41.jpg?width=640&crop=smart&auto=webp&s=5263324b68ccecb210b1de837c3ffb56d2b81d65 on Fri Sep  5 01:43:14 UTC 2025
