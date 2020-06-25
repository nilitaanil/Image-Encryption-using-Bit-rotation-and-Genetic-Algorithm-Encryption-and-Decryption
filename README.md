# Image-Encryption-using-Bit-rotation-and-Genetic-Algorithm-Encryption-and-Decryption
An Image Encryption Algorthim Using genetic Algorithm and Bit Slice and Rotation. Here We make use of the main feature of the Genetic Algorithm, crossover and mutation. Genetic Algorithm Acts as the main Encryption Algorithm.To increase the randomness and the security the bits(pixels) are sliced  in to 8  parst and then rotated  270 ,180 and 90.



Algorthim:
1. Read the image.
Encryption:
1. Apply crossover and mutation.
2. Apply Bitslice and rotation
Decryption:
1. Reverse the rotated bits 
2. Apply Genetic Algorithm.

### Image folder : 
consits of image.csv file which contains all the image names that are present in the image folder itself.
### image_analysis.csv : 
consists of the results after encrypting and decrypting the images. It includes the encryption and decryption time, SSIM value for encryption and decryption for each of the images.
### image_crypt.py : 
consists of the GA algorithm (file not used just for reference)
### bit_slice2.py : 
consists of the bitslice and rotation algorithm (file not used just for reference)
### bit_and_algo.py : 
code that should be excuted for this project. In this file, image_crypt.py and bit_slice2.py codes are combined
### output_encrypted.jpg: 
gives the encypted image obtained by executing bit_and_algo.py
### out_final.jpg: 
gives the decypted image obtained by executing bit_and_algo.py


