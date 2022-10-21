import tensorflow as tf
import module.keras_model as keras_module
            
            

# model = keras_module.load_model("./model/model_stft.hdf5")

denoise_model = tf.saved_model.load("./model/DTLN_norm_500h_saved_model")
# infer = denoise_model.signatures["serving_default"]
inference_func = denoise_model.signatures["serving_default"]
# model.summary()
print(inference_func)
# denoise_model