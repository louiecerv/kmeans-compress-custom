#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)

new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors = new_colors, title='Reduced color space: 16 colors')

logo_recolored = new_colors.reshape(logo.shape)

fig, ax = plt.subplots(1, 2, figsize=(16,6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(logo)
ax[0].set_title('Original Image', size = 16)
ax[1].imshow(logo_recolored)
ax[1].set_title('16-color image', size=16)

#save to file
# Save the image to file
reduced = '/content/drive/MyDrive/Colab Notebooks/CCS229/K Means/wvsu_logo_reduced.jpg'
plt.imsave(reduced, logo_recolored)

#show the compression ratio
# Get file size of image
orig_size = os.path.getsize(original)
reduced_size = os.path.getsize(reduced)

compression = round(reduced_size/orig_size, 2)
print(f'image size is reduce to {compression} of the original image.')

#run the app
if __name__ == "__main__":
    app()
