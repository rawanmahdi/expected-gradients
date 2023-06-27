#%%
## CODE FROM https://github.com/slundberg/shap
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np
import json
import shap
#%%
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#%%
# load pre-trained model and choose two images to explain
model = VGG16(weights='imagenet', include_top=True)
#%%
IMAGE_PATH0= '../img/meso-caoch.jpg'
IMAGE_PATH1= '../img/meso-grass.jpg'
image0 = tf.keras.preprocessing.image.load_img(IMAGE_PATH0, target_size=(224, 224))
img0 = tf.keras.preprocessing.image.img_to_array(image0)
image1 = tf.keras.preprocessing.image.load_img(IMAGE_PATH1, target_size=(224, 224))
img1 = tf.keras.preprocessing.image.img_to_array(image1)
cut=np.ndarray(img0)
cut0 = np.ndarray(img1)
X = np.ndarray(cut)
to_explain = X

#%%
XX,y = shap.datasets.imagenet50()
XX = XX[[29,31]]
#%%
# load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

#%%
# explain how the input to the 7th layer of the model explains the top two classes
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)
e = shap.GradientExplainer(
    (model.layers[7].input, model.layers[-1].output),
    map2layer(X, 7),
    local_smoothing=0 # std dev of smoothing noise
)
shap_values,indexes = e.shap_values(map2layer(to_explain, 7), ranked_outputs=2)
#%%
# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)
# plot the explanations
shap.image_plot(shap_values, to_explain, index_names)
# %%
