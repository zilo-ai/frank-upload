# -*- coding: utf-8 -*-
"""
    :author: Grey Li <withlihui@gmail.com>
    :copyright: (c) 2017 by Grey Li.
    :license: MIT, see LICENSE for more details.
"""
import os

from flask import Flask, render_template, request
from flask_dropzone import Dropzone
import matplotlib.pyplot as plt

import ProcessEKG_p3 as pEKG

# This will break when you use NGINX or when flask isn't single-threaded
# change this for the server
EKG = pEKG.ProcessEKG('/FIR/MBR/EKG/iOS_backend/ecg_ios/static/text_masks/',isSmall=True, doTransform=True, doFilter=True, doRotate=True)


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=30,
    DROPZONE_MAX_FILES=30,
)

dropzone = Dropzone(app)




@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')

        img = plt.imread(f)
        #f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))


        EKG.process_image(fobj=img)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
