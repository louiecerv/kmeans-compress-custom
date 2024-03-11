#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# Define the Streamlit app
def app():
    image_bytes = 0
    st.subheader('Upload an image for the compression task.')
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    if uploaded_file is not None:
        # Convert uploaded file to bytes
        image_bytes = uploaded_file.read()
        # Display uploaded image
        st.image(image_bytes, caption="Uploaded Image")

    original_image = np.array(image_bytes)

    fig, ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(original_image)
    st.pyplot(fig)

#run the app
if __name__ == "__main__":
    app()
