from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import json
from waitress import serve

from io import BytesIO

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('plant_disease.h5')
img_width, img_height = 224, 224
@app.route("/", methods=["GET"])
def welcome():
    return jsonify({
        "message":"Welcome to Plant Disease Detection System"
    })
@app.route("/", methods=["POST"])

def predict():
    classes={
    0: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    1: "Corn_(maize)___Common_rust_",
    2: "Corn_(maize)___Northern_Leaf_Blight",
    3: "Corn_(maize)___healthy",
    4: "Potato___Early_blight",
    5: "Potato___Late_blight",
    6: "Potato___healthy",
    7: "Soybean___healthy",
    8: "Tomato___Bacterial_spot",
    9: "Tomato___Early_blight",
    10: "Tomato___Late_blight",
    11: "Tomato___Leaf_Mold",
    12: "Tomato___Septoria_leaf_spot",
    13: "Tomato___Spider_mites Two-spotted_spider_mite",
    14: "Tomato___Target_Spot",
    15: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    16: "Tomato___Tomato_mosaic_virus",
    17: "Tomato___healthy"
    }
    
    # if 'file' not in request.files:
    #     return jsonify({"error": "No file part"})
    # if request.files.get("file").filename == '':
    #     return jsonify({"error": "No selected file"})
    
    files = request.files.getlist('file')
    # classes=json.loads(classes)
    file=files[0]
    img_width,img_height=224,224
    pil_img=Image.open(BytesIO(file.read()))
    img_array = np.array(pil_img).copy()
    print(img_array)
    img_array = np.resize(img_array,new_shape=(img_width, img_height,3))
    img_array = np.expand_dims(img_array, axis=0)
    logits = model.predict(img_array)[0]
    probabilities_tensor = tf.nn.softmax(logits)
    probabilities=list(probabilities_tensor)
    highest_probability = np.max(probabilities)
    predicted_class = probabilities.index(highest_probability)
    status=''
    treatment_available=False
    disease_populality="0/100"
    infected = "healthy" not in classes[predicted_class].split("___")[1].split(" ")
    diseases=[]
    if  infected:
        diseases=[x for x in classes[predicted_class].split("___")[1].split(" ")]
        treatment_available=True
        disease_populality="1/100"
    crop= classes[predicted_class].split("_")[0]
    accuracy="{:.2f}%".format(np.round(highest_probability*100,2),2)

 
    
    json_data = json.dumps({
        "crop": crop,
        "infected":infected,
        "diseases": diseases,
        "accuracy":accuracy,
        "treatment_available":treatment_available,
        "disease_populality":disease_populality,
        "recommendation":'''
        Congratulations! Your plant has been detected as uninfected. This is a positive sign of a healthy plant. To ensure the continued well-being of your crop, consider the following steps:
        Regular Inspection: Continue monitoring your plants regularly for any signs of disease or pest infestation. Early detection can help prevent problems from developing.

        Watering and Nutrition: Maintain a consistent watering schedule and provide the necessary nutrients to support optimal growth. Healthy plants are more resilient to potential threats.

        Crop Rotation: If applicable, practice crop rotation to prevent soil-borne diseases and pests from building up in the same area.

        Good Hygiene: Keep the growing area clean and free from debris that might harbor pests or diseases.

        Beneficial Insects: Encourage the presence of beneficial insects, such as ladybugs or predatory mites, which can help control pest populations naturally.

        Proper Pruning: Regularly trim any dead or diseased plant parts to prevent the spread of potential problems.

        Stay Informed: Stay updated on local weather conditions and any emerging plant health issues in your area.
        ''',
        }, indent=4, sort_keys=False) 

    response = app.response_class(
        response=json_data,
        status=200,
        mimetype='application/json'
    )
    return response
  
 
    
   

   

    # # Load your class label mapping here
    # files=request.files.getlist("file")
    # print(files)
    # if predicted_class in class_label_mapping:
    #     class_name = class_label_mapping[predicted_class]
    #     return jsonify({"crop": class_name, "accuracy": highest_probability * 100})
    # else:
    #     return jsonify({"crop": "unknown", "accuracy": 0.0})
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
