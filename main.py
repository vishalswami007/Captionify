import os
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import load_model
from flask import Flask, render_template, flash, redirect
from flask import request
from werkzeug.utils import secure_filename
import pickle
import numpy as np
from PIL import Image

img = ''

app = Flask(__name__)
app.secret_key = "secret key"
app.config['uploads'] = "static/uploads"

# load features from pickle
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()

mapping = {}
# process lines
for line in captions_doc.split('\n'):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc.,
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption


clean(mapping)
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in all_captions)


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text





@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/generate", methods=["POST", "GET"])
def generate_caption():
    global img
    vgg_model = VGG16()
    # restructure the model
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    image_path = img
    # load image
    print(img)
    image = load_img(image_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    feature = vgg_model.predict(image, verbose=0)
    # predict from the trained model
    model = load_model('best_model.h5')
    p_text = predict_caption(model, feature, tokenizer, max_length)
    return render_template("final_page.html", image=img, text1=p_text[9:-6])


@app.route("/submit-data", methods=["POST", "GET"])
def upload_image():
    global img
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect('index.html')
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['uploads'], filename))
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        img = 'static/uploads/' + filename
        return render_template("generate_page.html", image=img)


if __name__ == "__main__":
    app.run(debug=True)
