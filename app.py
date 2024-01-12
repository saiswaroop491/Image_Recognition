from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load your model here
cnn = tf.keras.models.load_model(r'C:\Users\SAI\MNK_Techfocus\dog_cat_recog.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                '',
                image_file.filename
            )
            image_file.save(image_location)
            test_image = tf.keras.utils.load_img(image_location, target_size=(64,64))
            test_image = tf.keras.utils.img_to_array(test_image)
            test_image = np.expand_dims(test_image,axis=0)
            result = cnn.predict(test_image)
            # Add your class indices here
            class_indices = {"cats": 0, "dogs": 1}

            dog_value = "[[0.48332196 0.51667804]]"

            result_list = np.array2string(result)

            if result_list == dog_value:
                predicted_class="Dog"
            else:
                predicted_class="Cat"
            return render_template('index.html', prediction=predicted_class)
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
