from flask import Flask, request, make_response, redirect, url_for, escape, render_template, Markup, session

import uuid
import sqlite3
import json
import os
# from threading import Thread
from multiprocessing import Process

import numpy as np

import librosa
from scipy.io import wavfile

from init_db import init_db
from hypercut import hypercut

UPLOAD_FOLDER = os.path.abspath(os.path.join(".", "uploads"))
DB_PATH = os.path.abspath("./cuts.db")

AUDIO_FILETYPE = "m4a"
TARGET_RATE = 16000

init_sql:str = "init.sql"

app = Flask(__name__)

def convert_frame_rate(path:str):
    data, sampling_rate = librosa.load(path)
    data = librosa.resample(data, sampling_rate, TARGET_RATE)
    wavfile.write(f"{path.split('.')[0]}.wav", TARGET_RATE, data)
    return data

def cut_thread(path:str):
    convert_frame_rate(path)
    print("entering hypercut")
    phrases, spaces = hypercut(f"{path.split('.')[0]}.wav")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("submitting to database")
    cur.execute("INSERT INTO cuts (cut_id, cut) VALUES (?, ?)", (os.path.basename(os.path.dirname(path)),json.dumps(dict(phrases=phrases, spaces=spaces))))

    cur.close()
    conn.commit()



@app.route('/upload', methods=['POST'])
def upload():
    response = dict(id=None, message="")

    try:
        f = request.files['audio']
    except:
        response['message'] = "No file provided"
        return make_response(response, 400)

    if not f:
        response['message'] = "No file provided"
        return make_response(response, 400)


    try:
        conn = sqlite3.connect(DB_PATH)
    except:
        response['message'] = "Internal server error: database failed to connect"
        return make_response(response, 500)

    cur = conn.cursor()
    guid = str(uuid.uuid4())
    while cur.execute("SELECT COUNT(*) FROM cut_ids WHERE cut_id = ?", (guid,)).fetchone()[0] > 0:
        guid = str(uuid.uuid4())

    cur.execute("INSERT INTO cut_ids (cut_id) VALUES (?)", (guid,))
    conn.commit()
    cur.close()

    os.makedirs(os.path.join(UPLOAD_FOLDER, guid))
    f.save(os.path.join(UPLOAD_FOLDER, guid, f"audio.{AUDIO_FILETYPE}")) # NOTE: use a self made file type
    thread = Process(target=cut_thread, args=(os.path.join(UPLOAD_FOLDER, guid, f"audio.{AUDIO_FILETYPE}"),))
    thread.start()

    print("responding now from /upload")
    return {'id': guid}

@app.route('/cut', methods=['GET'])
def cut():

    phrases = []
    spaces = []
    response = dict(phrases=phrases, spaces=spaces, message="")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn = sqlite3.connect(DB_PATH)
    except:
        response['message'] = "Internal server error: database failed to connect"
        return make_response(response, 500)

    cut_id = request.args.get('id', None, type=str)
    if cut_id is None:
        response['message'] = "No ID provided"
        return make_response(response, 400)

    if request.args.get('debug', 0, type=int):
        response['phrases'].append(
            {
                "start":1.2,
                "end":4.5,
                "priority": 0,
                "words":[
                    {
                        "start": 1.2,
                        "end": 2.3,
                        "word": "Hi"
                    },
                    {
                        "start": 2.4,
                        "end": 4.5,
                        "word": "Mikey"
                    }
                ]
            }
        )

        return response

    cur = conn.cursor()

    # check if the cut id provided exists
    # print(cur.execute("SELECT COUNT(*) FROM cut_ids where cut_id = ?", (cut_id,)).fetchone())
    if cur.execute("SELECT COUNT(*) FROM cut_ids where cut_id = ?", (cut_id,)).fetchone()[0] == 0:
        response['message'] = "Not a valid ID"
        cur.close()
        return make_response(response, 406)

    if cur.execute("SELECT COUNT(*) FROM cuts where cut_id = ?", (cut_id,)).fetchone()[0] == 0:
        response['message'] = "Not completed yet"
        cur.close()
        return make_response(response, 200)

    data = json.loads(cur.execute("SELECT cut FROM cuts WHERE cut_id = ?", (cut_id,)).fetchone()[0])
    response['phrases'] = data['phrases']
    response['spaces'] = data['spaces']

    cur.close()
    with open('test_cut_response.json', 'w') as f:
        json.dump(response, f, indent=2)
    return make_response(response, 200)

if __name__ == "__main__":
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
        print("Upload folder initialized at:")
        print(UPLOAD_FOLDER)

    if not os.path.isfile(DB_PATH):
        init_db(DB_PATH, init_sql)
        print("SQLite database initialize at:")
        print(DB_PATH)

    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)