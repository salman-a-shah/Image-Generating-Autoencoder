"""
Save models to saved_models folder
"""
def save_model(model, model_name):
    model_json = model.to_json()
    with open("saved_models\\" + model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("saved_models\\" + model_name+ ".h5")
    print("Saved", model_name, "to disk")

"""
Loads models from saved_models folder
"""
def load_model(model_name):
    model = ""
    try: 
        from keras.models import model_from_json 
        
        # Load model
        json_file = open('saved_models\\' + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    
        # Load weights
        model.load_weights('saved_models\\' + model_name + '.h5')
        print(model_name, 'loaded successfully')
    except:
        print("Failed to load model")
    return model

"""
Defines a Gaussian sampler to extract samples from latent space
"""
def get_gaussians(data, labels):
    
    import numpy as np
    m, n = data.shape
    if n == 0 or m == 0 or len(labels) != m:
        return None
    cov = np.zeros((10, n, n), dtype=float)
    means = np.zeros((10, n), dtype=float)
    counts = np.zeros(10, dtype=int)


    for i in range(m):
        j = labels[i]
        counts[j] +=1
        means[j] += data[i]
    for k in range(10):
        if counts[k] == 0:
            counts[k] = 1
        means[k] /= counts[k]
    for i in range(m):
        j = labels[i]
        cov[j] += np.outer(data[i]-means[j], (data[i].T-means[j].T))
    for k in range(10):
        cov[k] /= counts[k]
    return means, cov

"""
Defines binary cross entropy loss
"""
def CrossEntropy(y_true, y_pred):
    import numpy as np
    return -((y_true*np.log(y_pred))+((1-y_true)*np.log(1-y_pred)))
    
def binarycrossentropy(y_true,y_pred):
  loss = 0.0
  for i,_ in enumerate(y_true):
    loss += CrossEntropy(y_true[i],y_pred[i])
  return loss