#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans

# Define the Streamlit app
def app():
    data = st.session_state.image_data
    kmeans = MiniBatchKMeans(16)
    kmeans.fit(data)

    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

    #plot_pixels(data, 'Reduced color space: 16 colors', new_colors)

    img_recolored = new_colors.reshape(data.shape)

    fig, ax = plt.subplots(1, 2, figsize=(16,6), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(st.session_state.original)
    ax[0].set_title('Original Image', size = 16)
    ax[1].imshow(img_recolored)
    ax[1].set_title('16-color image', size=16)
    st.pyplot(fig)


    #show the compression ratio
    width, height = data.size
    # Get mode (e.g., RGB, RGBA) and corresponding bytes per pixel
    mode = data.mode
    bpp = {
        '1': 1, 'L': 8, 'P': 8, 'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24, 'I': 32, 'F': 32
    }[mode]

    # Calculate the total number of bytes in memory
    memory_size = width * height * bpp // 8

    st.write('original size =' + str(memory_size))

    fig, ax = plt.subplots()
    # Remove ticks from both axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img_recolored)
    st.pyplot(fig)

    #compression = round(reduced_size/orig_size, 2)
    #print(f'image size is reduce to {compression} of the original image.')

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
