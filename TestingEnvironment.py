"""
Import models
"""
from custom_functions import load_model

autoencoder = load_model("autoencoder")
encoder = load_model("encoder")
decoder = load_model("decoder")

# Import dataset
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
(x_train, train_labels), (x_test, test_labels) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # number of digits to display
fig = plt.figure(figsize=(10, 2))
fig.suptitle("Sample Reconstructions", fontweight='bold')
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

"""
Sample using single gaussians centered at sample means for each digit
"""
from custom_functions import get_gaussians  
means, covs = get_gaussians(encoded_imgs,test_labels)

n = 10
rows = 5
fig = plt.figure(figsize=(10,rows))
fig.suptitle("Images Sampled Using Gaussian Distributions Centered at Sample Means", fontweight='bold')
for i in range(n):
    new_encoded_imgs = np.random.multivariate_normal(means[i],0.2*covs[i],10)
    gaussian_samples = decoder.predict(new_encoded_imgs)
    for j in range(rows):
        ax = plt.subplot(rows,n, i+1+j*n)
        plt.imshow(gaussian_samples[j].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()  

"""
Sample using gaussian mixtures
Parameters intiialized through kmeans
"""
import numpy as np
from sklearn.mixture import GaussianMixture
#from custom_functions import binarycrossentropy

encoded_classified_dict={}
for i in range(10):
    ind=np.reshape(np.argwhere(test_labels==i),(np.argwhere(test_labels==i).shape[0]))
    encoded_classified_dict['{}'.format(i)]=encoded_imgs[ind,:]

n = 10
rows = 5
n_gaussians = 125
fig = plt.figure(figsize=(10,5))
fig.suptitle("Images Sampled Using Gaussian Mixtures", fontweight='bold')
for i in range(10):
    clf = GaussianMixture(n_components=n_gaussians,max_iter=1000,init_params="kmeans")
    clf.fit(encoded_classified_dict[str(i)])
    new_encoded_imgs,_ = clf.sample(n)
    new_decoded_imgs = decoder.predict(new_encoded_imgs)
    for j in range(rows):
        ax = plt.subplot(rows, n, i + 1 + j*n)
        plt.imshow(new_decoded_imgs[j].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()


      
"""
Standard Basis Vectors
"""
n=32
new_encoded_imgs = 10*np.identity(32)
decoded_imgs = decoder.predict(new_encoded_imgs)
fig = plt.figure(figsize=(8,4))
fig.suptitle("Standard Basis Vectors", fontweight='bold')
for i in range(n):
  ax = plt.subplot(4, 8, i + 1)
  plt.imshow(decoded_imgs[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

"""
Effects of having sparse means
"""
nonzeros = 2 # 0 to 31
sparse_means = means.copy()
for i, mu in enumerate(means):
  m = sorted(mu,reverse=True)
  for j, val in enumerate(mu):
    if m[nonzeros] > val:
      sparse_means[i][j] = 0

n=10
fig = plt.figure(figsize=(10, 2))
fig.suptitle("Sparse Means Effects")
new_encoded_imgs = means
decoded_imgs = decoder.predict(new_encoded_imgs)
for i in range(10):
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(decoded_imgs[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

new_encoded_imgs = sparse_means
decoded_imgs = decoder.predict(new_encoded_imgs)
for i in range(10):
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()


"""
Convex Combination of Sample Means
"""
from custom_functions import get_gaussians  
means, covs = get_gaussians(encoded_imgs,test_labels)
fps = 10
fig = plt.figure(figsize=(10,10))
fig.suptitle("Convex Combinations of Sample Means", fontweight='bold')
for i in range(10):
    s = means[i%10]     # source vector
    t = means[(i+1)%10] # target vector
    d = t - s           # direction vector
    for j in range(fps):
        new_vec = s + (j/fps) * d
        ax = plt.subplot(10, 10, i*10 + j + 1)
        plt.imshow(decoder.predict(new_vec.reshape(1,32)).reshape(28,28))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
plt.show()

## Code for gif creation
#fps = 15
#for i in range(10):
#    s = means[i%10]     # source vector
#    t = means[(i+1)%10] # target vector
#    d = t - s           # direction vector
#    for j in range(fps):
#        new_vec = s + (j/fps) * d
#        plt.figure(figsize=(3,3))
#        plt.imshow(decoder.predict(new_vec.reshape(1,32)).reshape(28,28))
#        plt.gray()
#        plt.xticks([])
#        plt.yticks([])
#        plt.savefig("figs\\" + str(i%10) + "to" + str((i+1)%10) + "frame" + str(j) + ".png", bbox_inches='tight')


"""
Principal Component Analysis
"""
encoded_classified_dict={}
for i in range(10):
    ind=np.reshape(np.argwhere(test_labels==i),(np.argwhere(test_labels==i).shape[0]))
    encoded_classified_dict['{}'.format(i)]=encoded_imgs[ind,:]
# eigenvalue/eigenvector analysis
fig = plt.figure(figsize=(6,4))
fig.suptitle("Eigenvalues of Encoded Sample Covariance Matrix", fontweight='bold')
encodedcovmat = np.cov(encoded_imgs.T)
vals, vecs = np.linalg.eig(encodedcovmat)
idx = vals.argsort()[::-1]
vals = vals[idx]
vecs = vecs[:,idx]
plt.bar(range(len(vals)), vals)
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue")
plt.show()

# comparing projections between 1 and 0 vs 4 and 9
fig = plt.figure(figsize=(10,4))
fig.suptitle("PCA 2D Projections - Comparing Spatial Differences", fontweight='bold')
testlists = [[1,0],[4,9]]
colors = ['blue', 'red', 'tab:green', 'orange']
for testlist in testlists:
    ax = plt.subplot(121 + testlists.index(testlist))
    for i in testlist:
        u1 = []
        u2 = []
        X = encoded_classified_dict[str(i)].T
        for j in range(X.shape[1]):
            u1.append(np.matmul(vecs[:,0],X[:,j]))
            u2.append(np.matmul(vecs[:,1],X[:,j]))
        ax.scatter(u1,u2,label=str(i), marker='o', alpha=0.6, color=colors.pop())
        ax.legend(loc="best",frameon=True,shadow=True,facecolor="w",edgecolor=(0,0,0))
plt.show()


"""
Using BIC to find the optimal number of Gaussians to use 
in our Guassian mixture model
Discarded
"""
#import numpy as np
#import itertools
#
#from scipy import linalg
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#
#from sklearn import mixture
#
#encoded_classified_dict={}
#for i in range(10):
#    ind=np.reshape(np.argwhere(test_labels==i),(np.argwhere(test_labels==i).shape[0]))
#    encoded_classified_dict['{}'.format(i)]=encoded_imgs[ind,:]
#
## Generate random sample, two components
#X = encoded_classified_dict[str(9)]
#
#lowest_bic = np.infty
#bic = []
#n_components_range = range(1, 26)
#cv_types = ['spherical', 'tied', 'diag', 'full']
#for cv_type in cv_types:
#    for n_components in n_components_range:
#        # Fit a Gaussian mixture with EM
#        gmm = mixture.GaussianMixture(n_components=n_components,
#                                      covariance_type=cv_type)
#        gmm.fit(X)
#        bic.append(gmm.bic(X))
#        if bic[-1] < lowest_bic:
#            lowest_bic = bic[-1]
#            best_gmm = gmm
#
#bic = np.array(bic)
#color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
#                              'darkorange'])
#clf = best_gmm
#bars = []
#
#fig = plt.figure(figsize=(8,8))
## BIC scores
#spl = plt.subplot(2, 1, 1)
#for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#    xpos = np.array(n_components_range) + .2 * (i - 2)
#    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
#                                  (i + 1) * len(n_components_range)],
#                        width=.2, color=color))
#plt.xticks(n_components_range)
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
#plt.title('BIC score per model')
#xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
#    .2 * np.floor(bic.argmin() / len(n_components_range))
#plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
#spl.set_xlabel('Number of components')
#spl.legend([b[0] for b in bars], cv_types)
#
#
#splot = plt.subplot(2, 1, 2)
#Y_ = clf.predict(X)
#for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                           color_iter)):
#    v, w = linalg.eigh(cov)
#    if not np.any(Y_ == i):
#        continue
#    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], alpha=0.8, marker='o', color=color)
#
#    # Plot an ellipse to show the Gaussian component
#    angle = np.arctan2(w[0][1], w[0][0])
#    angle = 180. * angle / np.pi  # convert to degrees
#    v = 2. * np.sqrt(2.) * np.sqrt(v)
#    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#    ell.set_clip_box(splot.bbox)
#    ell.set_alpha(.5)
#    splot.add_artist(ell)
#
#plt.xticks(())
#plt.yticks(())
#plt.title('Selected GMM: full model, 2 components')
#plt.subplots_adjust(hspace=.35, bottom=.02)
#plt.show()
