#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Define the Streamlit app
def app():


    text = """K-Mean Clustering as Image Compressor"""
    st.subheader(text)

    # Use session state to track the current form


    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    #st.image('MNIST.png', caption="Replace he image and replace this text with the description""")

    text = """Importing the K-means Clustering Algorithm: 
    \nfrom sklearn.cluster import MiniBatchKMeans: 
    This line imports a specific implementation of K-means called 
    MiniBatchKMeans from the scikit-learn library. 
    This variant is often more efficient for large datasets.
    \nCreating a K-means Model:
    \nkmeans = MiniBatchKMeans(16): This line creates a K-means 
    model with 16 clusters. It means the algorithm will group 
    the image's pixel colors into 16 representative colors.
    \nFitting the Model to Image Data:\
    kmeans.fit(data): This line applies the K-means algorithm to 
    the image data, which is likely represented as an array of 
    pixel colors. The algorithm iteratively groups the colors 
    into clusters based on their similarity.
    \nRecoloring the Image with Cluster Centroids:
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]: 
    This line creates a new array of colors for the image. It does this by:
    kmeans.predict(data): Assigning each pixel to its closest cluster centroid.
    kmeans.cluster_centers_: Using the actual colors of those centroids as the new pixel colors.
    \nVisualizing the Compressed Image:
    plot_pixels(data, colors=new_colors, title='Reduced color space: 16 colors'): 
    This line (presumably from a custom plotting function) displays the image using 
    the reduced set of 16 colors. You'll likely see a slightly less detailed, 
    but visually similar image.
    \nCreating a Recolored Image Array:
    flower_recolored = new_colors.reshape(flower.shape): This line 
    reshapes the array of new colors to match the original image's
    dimensions, creating a complete recolored image array that 
    can be saved or further processed.
    \nK-means reduces the number of colors in the image to 16, a form of compression.
    Visually, compression results in a less detailed but recognizable image.
    MiniBatchKMeans processes data in batches, making it efficient for large images."""
    st.write(text)
    st.write('Click on Introduction in the sidebar to start.')  

    
#run the app
if __name__ == "__main__":
    app()
