import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from PIL import Image

#Q1
# group_A = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15, 40, 45, 50, 62]
# group_B = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]

# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.boxplot(group_A, patch_artist=True)
# plt.title('Group A')
# plt.ylabel('Measurement Values')

# plt.subplot(1, 2, 2)
# plt.boxplot(group_B, patch_artist=True)
# plt.title('Group B')

# plt.suptitle('Separate Box Plots for Group A and Group B')
# plt.show()


# #Q2
# with open('genome.txt', 'r') as file:
#     content = file.read()
# genome_sequence = content
# print(genome_sequence)
# print(len(genome_sequence))

# t = np.linspace(0, 4 * np.pi, len(genome_sequence))
# x = np.cos(t)
# y = np.sin(t)
# z = np.linspace(0, 5, len(genome_sequence))

# coordinates = np.column_stack((x, y, z))
# colors = np.random.choice(['red', 'green', 'blue', 'orange', 'black', 'purple'], len(genome_sequence))

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x, y, z, c=colors)
# plt.title('3D Scatter with Random Colors')
# plt.show()

#Q3
img = Image.open('Breaking-Bad.jpg')
numpydata = np.asarray(img)

print(numpydata)

plt.imshow(numpydata)
plt.axis('off')
plt.title('Image Visualization with Matplotlib')
plt.show()

rot_img = np.rot90(numpydata)
plt.imshow(rot_img)
plt.axis('off')
plt.show()

flip_img = np.fliplr(numpydata)
plt.imshow(flip_img)
plt.axis('off')
plt.show()

# Grayscale conversion formula: Y = 0.299*R + 0.587*G + 0.114*B
gray_img = np.dot (numpydata[..., :3], [0.299, 0.587, 0.114])

plt.imshow(gray_img)
plt.axis('off')
plt.show()

#Q4
