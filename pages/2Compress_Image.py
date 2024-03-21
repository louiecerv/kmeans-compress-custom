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

    new_colors = kmeans.cluster_centers_[kmeans.predict(normalized_data)]

    plot_pixels(normalized_data, 'Reduced color space: 16 colors', new_colors)

    #convert the array to an image
    img_recolored = new_colors.reshape(normalized_data.shape)

    compressed = Image.new('RGB', original.size)
    compressed.putdata(img_recolored)


    fig, ax = plt.subplots(1, 2, figsize=(16,6), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(original)
    ax[0].set_title('Original Image', size = 16)
    ax[1].imshow(new_colors)
    ax[1].set_title('16-color image', size=16)
    st.pyplot(fig)
   
    width, height = original.size
    # Get mode (e.g., RGB, RGBA) and corresponding bytes per pixel
    mode = original.mode
    st.write(mode)
    bpp = {
        '1': 1, 'L': 8, 'P': 8, 'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24, 'I': 32, 'F': 32
    }[mode]
    # Calculate the total number of bytes in memory
    orig_size = width * height * bpp // 8
    st.write('original size =' + str(orig_size))

    width, height = compressed.size    
    mode = compressed.mode
    st.write(mode)
    bpp = {
        '1': 1, 'L': 8, 'P': 8, 'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24, 'I': 32, 'F': 32
    }[mode]
    reduced_size = width * height * bpp // 8
    st.write('compressed size =' + str(reduced_size))

    compression = round(reduced_size/orig_size, 2)
    st.Write(f'image size is reduce to {compression} of the original image.')

    st.subheader('The compressed image')
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
