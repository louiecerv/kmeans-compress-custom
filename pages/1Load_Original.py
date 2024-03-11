#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# Define the Streamlit app
def app():
    image = []

    if "original" not in st.session_state:
        st.session_state.original = []
    if "image_data" not in st.session_state:
        st.session_state.image_data = []

    st.subheader('Upload an image for the compression task.')
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        # Convert to RGB format (if necessary) for compatibility with st.image
        image = image.convert('RGB') if image.mode != 'RGB' else image
        st.session_state.original = image
        # Display the image with an informative caption
        #st.image(image, caption="Uploaded Image", use_column_width=True)

        original_image = np.array(image)

        fig, ax = plt.subplots()
        # Remove ticks from both axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(original_image)
        st.pyplot(fig)

        # Get dimensions of the image
        height, width, channels = original_image.shape

        data = original_image/255.0
        data = data.reshape(height * width, 3)
        st.session_state.image_data = data
        st.write(data.shape)

        plot_pixels(data, title= 'Input color space: 16 million possible colors')


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
