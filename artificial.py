import sys

import numpy as np
import matplotlib as mpl

from ml.circularAE import Autoencoder
from ml import DataPrep

# load data
trainname = "data/main1_x1-y2_train.csv"
testname = trainname.replace("train", "test")
data1, meta_data1 = DataPrep.load_data(trainname)
data2, meta_data2 = DataPrep.load_data(testname)

yCols_train = []
for i, label in enumerate(meta_data1['labels']):
    if 'y' in label:
        yCols_train.append(i)
yCols_test = []
for i, label in enumerate(meta_data2['labels']):
    if 'y' in label:
        yCols_test.append(i)
    if label == 'label':
        testCol = i

data_ = data1[:, yCols_train]
data_test = data2[:, yCols_test]
n, p = data_.shape

np.random.seed(7)
perm = np.arange(n)
np.random.shuffle(perm)
np.random.seed()
data_train = data_[perm[:int(0.8*n)]]
data_valid = data_[perm[int(0.8*n):]]
n, _ = data_train.shape

######## Hyperparameter ###########
redux = 1
layers = [128, 128]

nTrain = 30000
batch_size = 256
learning_rate = 0.001
factor_weights = 1.0
activations = 'tanh'
decoder_activations = ['tanh' for _ in range(len(layers))]
# decoder_activations[0] = 'lin'
last_layer = 'lin'

print_step = 1000
test_step = 1000

load = 0
train = 1
save = 1
savesave = 0
###################################


# Autoencoder
batch_gen = DataPrep.fast_random_inputbatch_generator(data_train, batch_size)

# init
ae_net = Autoencoder(None, p, redux, layers=layers, learning_rate=learning_rate, last_layer=last_layer,
					 factor_weights=factor_weights, activations=activations, decoder_activations=decoder_activations)

# train
if train:
    ae_net.train(nTrain, batch_gen, data_valid, test_step=test_step, print_step=print_step, count_tests=True,
                 plot=False, save=save)
if savesave:
    ae_net.save_model("out/model_"+filename)

# inference
enc_valid, dec_valid, rec_valid = ae_net.project_data(data_valid)
enc_test, dec_test, rec_test = ae_net.project_data(data_test)
min_enc, max_enc = np.min(enc_test), np.max(enc_test)
print("Min={min}, Max={max}".format(min=min_enc, max=max_enc))


# inference
# 2D -> 1D
if redux == 1 and p == 2:

    mpl.use("agg")
    import matplotlib.pyplot as plt
    latentSpace = np.reshape(np.linspace(min_enc, max_enc, 200),(-1,1))
    latentSpace_all = np.reshape(np.linspace(0, 2*np.pi, 200),(-1,1))
    latentSpace_dec = ae_net.decode(latentSpace)
    latentSpace_all_dec = ae_net.decode(latentSpace_all)

    # grid and arrows to where they are projected
    NN = 200
    deltax = np.max(data_test[:,0]) - np.min(data_test[:,0])
    xmin = np.min(data_test[:,0]) - 0.1*deltax
    xmax = np.max(data_test[:,0]) + 0.1*deltax
    deltay = np.max(data_test[:,1]) - np.min(data_test[:,1])
    ymin = np.min(data_test[:,1]) - 0.1*deltax
    ymax = np.max(data_test[:,1]) + 0.1*deltax
    xspace = np.linspace(xmin, xmax, NN)
    yspace = np.linspace(ymin, ymax, NN)
    grid = np.stack(np.meshgrid(xspace, yspace))
    grid = np.reshape(grid, (2,-1)).transpose()
    enc_grid, dec_grid, rec_grid = ae_net.project_data(grid)

    grid_error = np.reshape(np.log(rec_grid),(NN,NN))
    contour_vals = list(np.log(np.percentile(rec_valid, [68.2689492, 100]))) #
    contour_vals += [contour_vals[-1]+1, contour_vals[-1]+2]

    interpolation = None # 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'

    plt.close("all")
    # _ = [plt.plot([grid[i,0], dec_grid[i,0]], [grid[i,1], dec_grid[i,1]], color='gray') for i in range(len(grid))]
    CS = plt.imshow(grid_error[::-1],  interpolation=interpolation, extent=(xmin,xmax,ymin,ymax), aspect='auto') #

    plt.scatter(data_test[:,0], data_test[:,1], s=2, color="silver")
    plt.plot(latentSpace_all_dec[:,0], latentSpace_all_dec[:,1], color="orange")
    plt.plot(latentSpace_dec[:,0], latentSpace_dec[:,1], color="darkred")
    plt.contour(xspace, yspace, grid_error, contour_vals)

    plt.colorbar(CS) #cax=cax
    plt.savefig("plots/oSpace.png", dpi=150)

if redux == 1:
    # hist:
    plt.close("all")
    plt.hist(enc_test)
    plt.savefig("plots/lSpace.png", dpi=150)

# inference
# 3d -> 2d
if redux == 2 and p == 3:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    NN = 50
    latent1d = np.linspace(0, 2*np.pi, NN)
    latentSpace_all = np.reshape(np.stack(np.meshgrid(latent1d, latent1d)), (2,NN*NN)).T
    
    latentSpace_dec = ae_net.decode(latentSpace_all)
    
    
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # manifold
    ax.plot_wireframe(np.reshape(latentSpace_dec[:,0],(NN,NN)),np.reshape(latentSpace_dec[:,1],(NN,NN)),np.reshape(latentSpace_dec[:,2],(NN,NN)),  color=(0.1,0.3,0.9,0.5))
    
    n = 1
    ax.scatter(data_valid[:,0], data_valid[:,1], data_valid[:,2], s=3, color='silver')
    
    plt.show()