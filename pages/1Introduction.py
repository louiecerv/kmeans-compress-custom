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
    st.subheader('Upload an image for the compression task.')
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        # Convert to RGB format (if necessary) for compatibility with st.image
        image = image.convert('RGB') if image.mode != 'RGB' else image

        # Display the image with an informative caption
        st.image(image, caption="Uploaded Image", use_column_width=True)

        original_image = np.array(image)

        fig, ax = plt.subplots()
        # Remove ticks from both axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(original_image)
        st.pyplot(fig)

#run the app
if __name__ == "__main__":
    app()
