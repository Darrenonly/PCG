import numpy as np
# import cv2
import matplotlib.pyplot as plt
image = np.load("../feat/N/New_N_193/New_N_193_0.npy")
plt.plot(image)
plt.show()
# for i in range(0,image.shape[0]):
#     plt.imshow(image[i])
#     # cv2.imwrite(str(i)+".png",image[i,:,:])
#     plt.show()
