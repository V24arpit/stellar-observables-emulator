import classica_observables
import numpy as np
import h5py 
import matplotlib.pyplot as plt

# to run this just simply replace the file location in your disk.
with h5py.File(r"file path", "r") as f:
    X_train= []
    Y_train = []
    X_test = []
    Y_test = []
    
    tracks = f["grid"]["tracks"]
    all_tracks = sorted(list(tracks.keys()))
    training_tracks = all_tracks[:1000]  # to choose the initial tracks. you can edit it to the numbers you want.
    testing_tracks = all_tracks[1000:1050] # to chooose the validation tracks. you can also edit that just by changing the numbers.

# to extract the desired 5 input parameters. it can be modified for any file just make sure mention their keys as it is as mentioned in the file
    for tr_track in training_tracks : 
        t = tracks[tr_track]
        x = np.column_stack([t["massini"][:],t["FeHini"][:],t["yini"][:],t["alphaMLT"][:],t["age"][:]])
        y = np.column_stack([t["Teff"][:],t["LPhot"][:],t["FeH"][:]])
        X_train.append(x)
        Y_train.append(y)

# same for validation tracks.    
    for te_track in testing_tracks:
        t = tracks[te_track]
        x = np.column_stack([t["massini"][:],t["FeHini"][:],t["yini"][:],t["alphaMLT"][:],t["age"][:]])
        y = np.column_stack([t["Teff"][:],t["LPhot"][:],t["FeH"][:]])
        X_test.append(x)
        Y_test.append(y)
        
X_train = np.vstack(X_train)
Y_train = np.vstack(Y_train)
X_test = np.vstack(X_test)
Y_test = np.vstack(Y_test)

# here X_scaled means X_scaled_training ,i.e. for training data set.
X_scaled = X_train.copy()
# taking log(base10) for each parameters except Fe/H.
X_scaled[:,0] = np.log10(X_scaled[:,0])
X_scaled[:,2] = np.log10(X_scaled[:,2])
X_scaled[:,3] = np.log10(X_scaled[:,3])
X_scaled[:,4] = np.log10(X_scaled[:,4])

Y_scaled = Y_train.copy()
Y_scaled[:,0] = np.log10(Y_scaled[:,0])
Y_scaled[:,1] = np.log10(Y_scaled[:,1])

x_mean = np.mean(X_scaled,axis=0)
x_std = np.std(X_scaled,axis=0)

y_mean = np.mean(Y_scaled,axis=0)
y_std = np.std(Y_scaled,axis=0)

# this is basically scaling .
X_scaled = (X_scaled - x_mean)/x_std
Y_scaled = (Y_scaled - y_mean)/y_std

# same goes for validation tracks. 
X_scaled_test = X_test.copy()
X_scaled_test[:,0] = np.log10(X_scaled_test[:,0])
X_scaled_test[:,2] = np.log10(X_scaled_test[:,2])
X_scaled_test[:,3] = np.log10(X_scaled_test[:,3])
X_scaled_test[:,4] = np.log10(X_scaled_test[:,4])

Y_scaled_test = Y_test.copy()
Y_scaled_test[:,0] = np.log10(Y_scaled_test[:,0])
Y_scaled_test[:,1] = np.log10(Y_scaled_test[:,1])

X_scaled_test = (X_scaled_test - x_mean)/x_std
Y_scaled_test = (Y_scaled_test - y_mean)/y_std

net = classica_observables.Class_obser([5,64,64,3])
'''previous model is loaded here. 
make sure to give location of the zip file in you disk.
make sure to comment it down for your very first run. as initially the file is emnpty.
afterwards you can remove the comment'''
net.load_model("path for zip file")   
print("Previous model loaded")

training_data = []
for i in range(len(X_scaled)):
    x = X_scaled[i].reshape(5,1)
    y = Y_scaled[i].reshape(3,1)
    training_data.append((x,y))

validation_data = []
for i in range(len(X_scaled_test)):
    x = X_scaled_test[i].reshape(5,1)
    y = Y_scaled_test[i].reshape(3,1)
    validation_data.append((x,y))

# here you edit epochs,mini_batch_size,eta 
train_y , val_y = net.sgd(training_data,validation_data,epochs=150,mini_batches_size=64,eta=0.001)
net.save_model("zip file path",x_mean,x_std,y_mean,y_std) # this will save the weight and biases

# graph plotting part is done here .
train_y = np.array(train_y)
val_y = np.array(val_y)
x_axis_list = []
for a in range(1,151):
    x_axis_list.append(a)
x_axis = np.array(x_axis_list)
plt.plot(x_axis,train_y,label = "Training data plot",color = "red")
plt.plot(x_axis,val_y,label =  "Validation data plot",color = "blue")
plt.scatter(x_axis,train_y,label= "Data point for training data",color = "yellow")
plt.scatter(x_axis,val_y,label = "Data points for Validation data",color ="green")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE VS Epochs Curve")
plt.legend()
plt.grid(True,alpha = 0.2)
plt.show()

# error plotting part is done here.
teff_error = []
lum_error = []
feh_error = []
model_no = []
# unscaling the data to calculate the realtive error
for i,(x,y) in enumerate(validation_data):
    pred = net.prediction(x)
    actual = y.flatten()
    pred = pred.flatten()

    actual = actual * y_std +y_mean
    pred = pred * y_std +y_mean

    actual_teff = 10**actual[0]
    pred_teff = 10**pred[0]

    actual_lum = 10**actual[1]
    pred_lum = 10**pred[1]

    actual_feh = actual[2]
    pred_feh = pred[2]

    err_teff = (actual_teff - pred_teff)/actual_teff
    err_lum = (actual_lum - pred_lum)/actual_lum
    err_feh = (actual_feh - pred_feh)

    teff_error.append(err_teff)
    lum_error.append(err_lum)
    feh_error.append(err_feh)
    model_no.append(i)


fig,ax = plt.subplots(2,2,figsize = (12,8))
ax[0,0].hist(teff_error,bins = 50,color= "royalblue",edgecolor="black")
ax[0,0].set_title(r"$T_{eff}$ Relative Error")
ax[0,0].set_xlabel("(Actual-Pred)/Actual")
ax[0,0].set_ylabel("Frequency")
ax[0,0].grid(True,alpha=0.2)

ax[0,1].hist(lum_error,bins= 50,color= "royalblue",edgecolor="black")
ax[0,1].set_title("Luminosity Relative Error")
ax[0,1].set_xlabel("(Actual-Pred)/Actual")
ax[0,1].set_ylabel("Frequency")
ax[0,1].grid(True,alpha =0.2)

ax[1,0].hist(feh_error,bins=50,color= "royalblue",edgecolor="black")
ax[1,0].set_title("Fe/H Relative Error")
ax[1,0].set_xlabel("(Actual-Pred)")
ax[1,0].set_ylabel("Frequency")
ax[1,0].grid(True,alpha=0.2)

ax[1,1].axis("off")
plt.tight_layout()
plt.show()
