#%%
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_explain.core.smoothgrad import SmoothGrad
#%%
#IMAGE_PATH= '../img/meso-grass.jpg'
IMAGE_PATH= '../img/meso-caoch.jpg'
model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
#%%
model.summary()
#%%
image = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(image)
data = ([img], None)
#%%
siamese_cat_class_index = 284
explainer = SmoothGrad()
# Compute SmoothGrad on VGG16
grid = explainer.explain(data, model, siamese_cat_class_index, 20, 1.)
#%%
# Save result
explainer.save(grid, '.', 'smoothgrad.png')
#%%
# Plot result
fig, ax = plt.subplots(1, 2, figsize=(16, 10))
ax[0].imshow(image)
ax[1].imshow(grid, cmap='gray', vmin=0, vmax=255)
plt.show()
# %%
