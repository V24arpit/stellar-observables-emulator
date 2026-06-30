import h5py
import numpy as np 
import matplotlib.pyplot as plt
import astro_seis
from sklearn.decomposition import PCA

# this file by default uses 1000 training tracks and 200 validation tracks.
# most of the structure is similar to classical observables.
with h5py.File(r"file path here","r") as f:
    X_train= []
    Y_train = []
    X_test = []
    Y_test = [] 
# this function basically extracts the frequency for radial order between 11 to 27 (both included) and for l=0,1,2
# and if the model dosent met this requirment then its discarded.
# you can edit this as your desire just change the numbers.
    def extrct_freq(track,i):
        freq = track["osc"][i,0]
        lval = track["osckey"][i,0]
        nval = track["osckey"][i,1]
        selected = []

        for l in [0,1,2]:
            data = ((lval == l)& (nval >= 11) & (nval <=27))
            if np.sum(data) != 17:
                return None
            freq_l = freq[data]
            n_l = nval[data]
            order = np.argsort(n_l)
            freq_l = freq_l[order]
            selected.extend(freq_l)
        return np.array(selected)
    
    tracks = f["grid"]["tracks"]
    all_tracks = sorted(list(tracks.keys()))
    training_tracks = all_tracks[:1000]   
    testing_tracks  = all_tracks[1000:1200]
    
    for track_name in training_tracks:
        t = tracks[track_name]
        model_num = len(t["modnum"])
        for i in range (model_num):
            freqs = extrct_freq(t,i)
            if freqs is None:
               continue
            x = np.array([t["massini"][i],t["FeHini"][i],t["yini"][i],t["alphaMLT"][i],t["age"][i]])
            X_train.append(x)
            Y_train.append(freqs)
    
    for track_name in testing_tracks:
        t = tracks[track_name]
        model_num = len(t["modnum"])
        for i in range (model_num):
            freqs = extrct_freq(t,i)
            if freqs is None:
               continue
            x = np.array([t["massini"][i],t["FeHini"][i],t["yini"][i],t["alphaMLT"][i],t["age"][i]])
            X_test.append(x)
            Y_test.append(freqs)
              
        
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(X_train.shape)
print(Y_train.shape)

# this block of code appies pca as there are 51 frequencies .
# so thats why pca is applied turning them to 8 pca coefficients.
pca = PCA(n_components=8)
Y_train_pca = pca.fit_transform(Y_train)
Y_test_pca = pca.transform(Y_test)


# X_scaled  = X_scaled_training 
X_scaled = X_train.copy()
X_scaled[:,0] = np.log10(X_scaled[:,0])
X_scaled[:,2] = np.log10(X_scaled[:,2])
X_scaled[:,3] = np.log10(X_scaled[:,3])
X_scaled[:,4] = np.log10(X_scaled[:,4])

x_mean = np.mean(X_scaled,axis=0)
x_std = np.std(X_scaled,axis=0)

y_mean = np.mean(Y_train_pca,axis=0)
y_std = np.std(Y_train_pca,axis=0)

X_scaled = (X_scaled - x_mean)/x_std
Y_scaled = (Y_train_pca - y_mean)/y_std


X_scaled_test = X_test.copy()
X_scaled_test[:,0] = np.log10(X_scaled_test[:,0])
X_scaled_test[:,2] = np.log10(X_scaled_test[:,2])
X_scaled_test[:,3] = np.log10(X_scaled_test[:,3])
X_scaled_test[:,4] = np.log10(X_scaled_test[:,4])


X_scaled_test = (X_scaled_test - x_mean)/x_std
Y_scaled_test = (Y_test_pca - y_mean)/y_std


net = astro_seis.Astro_seis([5,128,128,128,128,128,128,8])
"""net.load_model("zip file path here")
print("Previous model loaded")"""

training_data = []
for i in range(len(X_scaled)):
    x = X_scaled[i].reshape(5,1)
    y = Y_scaled[i].reshape(8,1)
    training_data.append((x,y))

validation_data = []
for i in range(len(X_scaled_test)):
    x = X_scaled_test[i].reshape(5,1)
    y = Y_scaled_test[i].reshape(8,1)
    validation_data.append((x,y))


train_y , val_y = net.sgd(training_data,validation_data,epochs=30,mini_batches_size=64,eta=0.001)
net.save_model("zip file path here",x_mean,x_std,y_mean,y_std)

# this block calculates error .
error_l0 = []
error_l1 = []
error_l2 = []
for x,y in validation_data:
    pred_scaled = net.prediction(x).flatten()
    pred_pca = pred_scaled * y_std + y_mean
    actual_pca = y.flatten() * y_std + y_mean
    pred_freq = pca.inverse_transform(pred_pca.reshape(1,-1))[0]
    actual_freq = pca.inverse_transform(actual_pca.reshape(1,-1))[0]

    error_l0.append((actual_freq[0]-pred_freq[0])/actual_freq[0])
    error_l1.append((actual_freq[17]-pred_freq[17])/actual_freq[17])
    error_l2.append((actual_freq[34]-pred_freq[34])/actual_freq[34])
    
# plotting part is done here .    
train_y = np.array(train_y)
val_y = np.array(val_y)
x_axis_list = []
for a in range(1,31):
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

fig,ax = plt.subplots(1,3,figsize=(15,5))

ax[0].hist(error_l0,bins = 50,color= "royalblue",edgecolor="black")
ax[0].set_title("Lowest radial order for l = 0")

ax[1].hist(error_l1,bins = 50,color= "royalblue",edgecolor="black")
ax[1].set_title("Lowest radial order for l = 1")

ax[2].hist(error_l2,bins = 50,color= "royalblue",edgecolor="black")
ax[2].set_title("Lowest radial order for l = 2")

for a in ax:
    a.set_xlabel("(Actual-Pred)/Actua")
    a.set_ylabel("Frequency")
    a.grid(True,alpha = 0.2)
plt.tight_layout()
plt.show()

