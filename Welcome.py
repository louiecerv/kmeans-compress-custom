#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Define the Streamlit app
def app():

    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

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
    plot_pixels(data, colors=new_colors, title='Reduced color space: 16 colors'): This line (presumably from a custom plotting function) displays the image using the reduced set of 16 colors. You'll likely see a slightly less detailed, but visually similar image.
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

 
    """
    # Extract only the specified number of images and labels
    size = 10000
    X, y = st.session_state.mnist
    X = X[:size]
    y = y[:size]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #save the values to the session state    
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test    
    """

    st.sidebar.subheader('Select the classifier')

    # Create the selection of classifier
    clf = DecisionTreeClassifier()
    options = ['Decision Tree', 'Random Forest Classifier', 'Extreme Random Forest Classifier', 'K-Nearest Neighbor']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option =='Random Forest Classifier':
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        st.session_state['selected_model'] = 1
    elif selected_option=='Extreme Random Forest Classifier':        
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)
        st.session_state['selected_model'] = 2
    elif selected_option == 'K-Nearest Neighbor':
        clf = KNeighborsClassifier(n_neighbors=5)
        st.session_state['selected_model'] = 3
    else:
        clf = DecisionTreeClassifier()
        st.session_state['selected_model'] = 0

    # save the clf to the session variable
    st.session_state['clf'] = clf
    
#run the app
if __name__ == "__main__":
    app()
