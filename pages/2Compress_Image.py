#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from PIL import Image

# Define the Streamlit app
def app():
    original = st.session_state.original
    original_data = st.session_state.original_data
    normalized_data = st.session_state.normalized_data

    kmeans = MiniBatchKMeans(16)
    kmeans.fit(normalized_data)
    st.subheader('Color reduction using K-Means Clustering')

    new_colors = kmeans.cluster_centers_[kmeans.predict(normalized_data)]

    plot_pixels(new_colors, 'Reduced color space: 16 colors')

    #convert the array to an image
    img_recolored = new_colors.reshape(original_data.shape)

    fig, ax = plt.subplots(1, 2, figsize=(16,6), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(original)
    ax[0].set_title('Original Image 16 Million colors', size = 16)
    ax[1].imshow(img_recolored)
    ax[1].set_title('16-color image', size=16)
    st.pyplot(fig)

    st.subheader('The compressed image')
    with st.expander("Show description"):
        text = """Fewer colors: The most prominent feature will be a significant 
        reduction in the number of colors displayed. The image will resemble a 
        painting with a limited palette.
        Blockiness or Posterization: Depending on the number of clusters (k) chosen, 
        the image might exhibit a blocky or posterized effect. Areas with 
        subtle color variations in the original image might appear more 
        uniform with a single dominant color from the chosen cluster.
        Loss of detail: Fine details and smooth color gradients might be 
        lost or appear less pronounced. The image might take on a more 
        cartoonish or flat look.
        Overall color representation: The dominant colors in the original 
        image will likely be preserved, but finer color nuances 
        will be sacrificed."""
        st.write(text)    
    fig, ax = plt.subplots()
    # Remove ticks from both axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img_recolored)
    st.pyplot(fig)

def plot_pixels(data, title, colors=None, N=1000):
    if colors is None:
        colors = data

        rng = np.random.RandomState(0)
        i = rng.permutation(data.shape[0])[:N]
        colors = colors[i]
        R, G, B = data[i].T

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].scatter(R, G, color=colors, marker='.')
        ax[0].set(xlabel='red', ylabel='green', xlim=(0, 1), ylim=(0, 1))
        ax[1].scatter(R, B, color=colors, marker='.')
        ax[1].set(xlabel='red', ylabel='blue', xlim=(0, 1), ylim=(0, 1))  
        fig.suptitle(title, size=20)
        st.pyplot(fig)

#run the app
if __name__ == "__main__":
    app()
