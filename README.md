# Image-Generating-Autoencoder
This was originally created as a class project for a machine learning course at UCLA by the following authors: Amaael Antonini, Krystian Galik, Salman Shah, Zixin Zhang. The files uploaded had been edited and organized by Salman Shah.

## Purpose
The purpose of the project was to build a machine learning model that can generate unseen handwritten digits. The architecture is simply an autoencoder with a sampling method implemented in the latent space. The two sampling methods we experimented with were Gaussian sampling and Gaussian mixture sampling. We also experimented with the autoencoder architecture, latent space visualization, convex combinations of sample means, and PCA.

<b><p align="center">Navigating the encoded 32-dim space from one sample mean to another</p></b>
<p align="center">
  <img width="128" height="128" src="https://raw.githubusercontent.com/salman-a-shah/Image-Generating-Autoencoder/master/figs/convex_combination.gif">
</p>

## Gaussian Mixture Sampling
<p align="center">
  <img width="906" height="451" src="https://raw.githubusercontent.com/salman-a-shah/Image-Generating-Autoencoder/master/figs/gaussian_mixture_samples.png">
</p>

## Principal Component Analysis
### 2D Projection
<p align="center">
  <img width="906" height="360" src="https://raw.githubusercontent.com/salman-a-shah/Image-Generating-Autoencoder/master/figs/pca_projections.png">
</p>
A 2D projection showing how visually similar images (ex: 4 and 9) have overlap in their clusters while visually different images (ex: 0 and 1) have clusters that are separated.

### Eigenvalues

<p align="center">
  <img width="600" height="398" src="https://raw.githubusercontent.com/salman-a-shah/Image-Generating-Autoencoder/master/figs/pca_eigenvalues.png">
</p>

Principal component eigenvalues showing that one could reasonably project the dataset down to 20 dimensions.

## Additional Details
A blog post describing the model in detail can be found [here](https://thephilosophersdomain.com/portfolio/autoencoder/).

See "image-generation-autoencoder.pdf" for a paper detailing our results.
