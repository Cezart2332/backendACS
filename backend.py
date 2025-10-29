import os
import statistics as st
import time
from io import StringIO, BytesIO
import base64
import cv2
import numpy as np
from numpy import linalg as la
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

from flask import Flask, send_file, make_response

import csv

poze = []
NR_PERSOANE = 40
NR_POZE_TOTALE = 10
A = None
T = None
NR_POZE_ANTRENARE = 0
NR_POZE_TESTARE = 0
PREPROCESSED = False
STATS_RUNNED = False
STATS_DATA = None

norme = ["Manhattan", "Euclidean", "Infinity", "Cosinus"]

def preprocessing(training_num):
    global A, T, NR_POZE_ANTRENARE, NR_POZE_TESTARE, PREPROCESSED, STATS_RUNNED, STATS_DATA
    NR_POZE_ANTRENARE = training_num
    NR_POZE_TESTARE = NR_POZE_TOTALE - NR_POZE_ANTRENARE
    print(f"Numar poze antrenare setat la: {NR_POZE_ANTRENARE}")
    print(f"Initializare matrice A de dimensiune: {10304} x {40*NR_POZE_ANTRENARE}")
    print(f"Initializare matrice T de dimensiune: {10304} x {40*NR_POZE_TESTARE}")
    A = np.zeros([10304, 40*NR_POZE_ANTRENARE])
    T = np.zeros([10304, 40*NR_POZE_TESTARE])
    path = "./att_faces"
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Folderul '{path}' nu exista. Verifica calea.")

    for i in range(1, NR_PERSOANE + 1):
        person_path = path + "/s" + str(i) + "/"
        for j in range(1, NR_POZE_ANTRENARE + 1):
            person_training_path = person_path + str(j) + ".pgm"
            training_photo = np.array(cv2.imread(person_training_path, 0)) # type: ignore
            training_photo = training_photo.reshape(
                10304,
            )
            A[:, NR_POZE_ANTRENARE * (i - 1) + (j - 1)] = training_photo

    for i in range(1, NR_PERSOANE + 1):
        person_path = path + "/s" + str(i) + "/"
        for j in range(NR_POZE_ANTRENARE + 1, NR_POZE_TOTALE + 1):
            person_test_path = person_path + str(j) + ".pgm"
            test_photo = np.array(cv2.imread(person_test_path, 0)) # type: ignore
            test_photo = test_photo.reshape(
                10304,
            )
            coloana = (NR_POZE_TESTARE) * (i - 1) + (j - (NR_POZE_ANTRENARE + 1))
            T[:, coloana] = test_photo
    PREPROCESSED = True
    STATS_RUNNED = False
    STATS_DATA = None

def nn(norm, p):
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing mai intai.")
    z = np.zeros(len(A[0]))
    for i in range(len(A[0])):
        if norm == "Manhattan":
            z[i] = la.norm(A[:, i] - p, 1)
        elif norm == "Euclidean":
            z[i] = la.norm(A[:, i] - p, 2)
        elif norm == "Infinity":
            z[i] = la.norm(A[:, i] - p, np.inf)
        elif norm == "Cosinus":
            z[i] = 1 - (np.dot(A[:, i], p)) / (la.norm(A[:, i], 2) * la.norm(p, 2))
        else:
            raise ValueError("Norma Invalida!")
    pozitia = np.argmin(z)
    return pozitia


def k_nn(norm, p, k):
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing mai intai.")
    z = np.zeros(len(A[0]))
    for i in range(len(A[0])):
        if norm == "Manhattan":
            z[i] = la.norm(A[:, i] - p, 1)
        elif norm == "Euclidean":
            z[i] = la.norm(A[:, i] - p, 2)
        elif norm == "Infinity":
            z[i] = la.norm(A[:, i] - p, np.inf)
        elif norm == "Cosinus":
            z[i] = 1 - (np.dot(A[:, i], p)) / (la.norm(A[:, i], 2) * la.norm(p, 2))
        else:
            raise ValueError("Norma Invalida!")
    indicii = np.argsort(z)[:k]
    pozitii = indicii // NR_POZE_ANTRENARE
    pozitia = st.mode(pozitii) * NR_POZE_ANTRENARE
    return pozitia
    


def statistics():
    global STATS_RUNNED, rata_recunoastere_text, timp_interogare_text, STATS_DATA
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing mai intai.")
    if STATS_RUNNED is True:
        raise Exception("Statistici deja calculate. Ruleaza preprocessing() cu alta configuratie pentru a reseta.")
    nr_total_teste = NR_PERSOANE * (10 - NR_POZE_ANTRENARE)
    STATS_DATA = {"nn": [], "k_nn": []}
    rata_recunoastere_text = "Statistici nn:\n"
    timp_interogare_text = "Statistici nn:\n"
    print("Statistici nn:")
    for norma in norme:
        print(f"Calcul statistici pentru norma: {norma}")
        nr_recunoasteri_corecte = 0
        timp_total_interogare = 0
        for j in range(len(T[0])):
            t0 = time.perf_counter()
            persoana_testata = j // NR_POZE_TESTARE
            persoane_cautata = nn(norma, T[:, j]) // NR_POZE_ANTRENARE
            t1 = time.perf_counter()
            timp_total_interogare += t1 - t0
            if persoana_testata == persoane_cautata:
                nr_recunoasteri_corecte = nr_recunoasteri_corecte + 1
        rr = nr_recunoasteri_corecte / nr_total_teste
        rata_procente = rr * 100
        tmi = timp_total_interogare / nr_total_teste
        STATS_DATA["nn"].append({"norm": norma, "rate": rata_procente, "time": tmi})
        print(f"Rata de recunoastere norma={norma}: {rata_procente:.2f}%")
        print(f"Timp mediu de interogare norma={norma}: {tmi:.8f}")
        rata_recunoastere_text += f"Rata de recunoastere norma={norma}: {rata_procente:.2f}%\n"
        timp_interogare_text += f"Timp mediu de interogare norma={norma}: {tmi:.8f}\n"

    rata_recunoastere_text += "Statistici k_nn:\n"
    timp_interogare_text += "Statistici k_nn:\n"
    print("Statistici k_nn:")
    for k in range(1, 9, 2):
        for norma in norme:
            nr_recunoasteri_corecte = 0
            timp_total_interogare = 0
            for j in range(len(T[0])):
                t0 = time.perf_counter()
                persoana_testata = j // NR_POZE_TESTARE
                persoane_cautata = k_nn(norma, T[:, j], k) // NR_POZE_ANTRENARE
                t1 = time.perf_counter()
                timp_total_interogare += t1 - t0
                if persoana_testata == persoane_cautata:
                    nr_recunoasteri_corecte = nr_recunoasteri_corecte + 1
            rr = nr_recunoasteri_corecte / nr_total_teste
            rata_procente = rr * 100
            tmi = timp_total_interogare / nr_total_teste
            STATS_DATA["k_nn"].append({"norm": norma, "k": k, "rate": rata_procente, "time": tmi})
            print(f"Rata de recunoastere norma={norma} si k={k}: {rata_procente:.2f}%")
            print(f"Timp mediu de interogare norma={norma} si k={k}: {tmi:.8f}")
            rata_recunoastere_text += f"Rata de recunoastere norma={norma} si k={k}: {rata_procente:.8f}%\n"
            timp_interogare_text += f"Timp mediu de interogare norma={norma} si k={k}: {tmi:.8f}\n"
            
    STATS_RUNNED = True

app = Flask(__name__)

# Enable CORS for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Cache-Control')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/statistics/export/<format_type>")
def export_statistics(format_type):
    """Export statistics as CSV or TXT file."""
    if PREPROCESSED is False:
        return {"error": "Datele nu au fost preprocesate."}, 400
    if STATS_RUNNED is False:
        statistics()
    if STATS_DATA is None:
        return {"error": "Statistici indisponibile."}, 500
    
    if format_type == 'txt':
        # Create TXT file
        output = StringIO()
        output.write("=== STATISTICI RECUNOASTERE FETE ===\n\n")
        output.write(rata_recunoastere_text)
        output.write("\n")
        output.write(timp_interogare_text)
        
        # Convert to bytes
        mem = BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        output.close()
        
        return send_file(
            mem,
            mimetype='text/plain',
            as_attachment=True,
            download_name='statistici_recunoastere.txt'
        )
    
    elif format_type == 'csv':
        # Create CSV from structured statistics data
        output = StringIO()
        writer = csv.writer(output)

        writer.writerow(['Algoritm', 'Norma', 'K', 'Rata Recunoastere (%)', 'Timp Mediu Interogare (s)'])

        for entry in STATS_DATA["nn"]:
            writer.writerow([
                'NN',
                entry['norm'],
                '-',
                f"{entry['rate']:.2f}",
                f"{entry['time']:.8f}"
            ])

        for entry in STATS_DATA["k_nn"]:
            writer.writerow([
                'k-NN',
                entry['norm'],
                entry['k'],
                f"{entry['rate']:.2f}",
                f"{entry['time']:.8f}"
            ])

        mem = BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        output.close()

        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name='statistici_recunoastere.csv'
        )

    else:
        return {"error": "Format invalid. Foloseste 'txt' sau 'csv'."}, 400


@app.route("/statistics/graph")
def statistics_graph():
    if PREPROCESSED is False:
        return {"error": "Datele nu au fost preprocesate."}, 400
    if STATS_RUNNED is False:
        statistics()
    if STATS_DATA is None:
        return {"error": "Statistici indisponibile."}, 500

    norms_order = [entry['norm'] for entry in STATS_DATA["nn"]]
    nn_by_norm = {entry['norm']: entry for entry in STATS_DATA["nn"]}

    k_values = sorted({1} | {entry['k'] for entry in STATS_DATA["k_nn"]})

    knn_lookup = {}
    for entry in STATS_DATA["k_nn"]:
        knn_lookup.setdefault((entry['norm'], entry['k']), entry)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')

    colors = plt.cm.viridis(np.linspace(0, 1, len(norms_order))) if norms_order else []

    for idx, norm in enumerate(norms_order):
        color = colors[idx] if idx < len(colors) else None
        rates_line = []
        times_line = []
        for k in k_values:
            if k == 1:
                nn_entry = nn_by_norm.get(norm)
                rates_line.append(nn_entry['rate'] if nn_entry else np.nan)
                times_line.append(nn_entry['time'] if nn_entry else np.nan)
            else:
                entry = knn_lookup.get((norm, k))
                rates_line.append(entry['rate'] if entry else np.nan)
                times_line.append(entry['time'] if entry else np.nan)

        axes[0].plot(k_values, rates_line, marker='o', label=f"{norm}", color=color)
        axes[1].plot(k_values, times_line, marker='o', label=f"{norm}", color=color)

    axes[0].set_title('Rata de recunoastere (%)')
    axes[0].set_ylabel('Procente (%)')
    axes[0].set_xticks(k_values)
    axes[0].set_xlabel('K-uri')
    axes[0].grid(axis='both', linestyle='--', alpha=0.3)

    axes[1].set_title('Timp mediu de interogare (s)')
    axes[1].set_ylabel('Secunde')
    axes[1].set_xticks(k_values)
    axes[1].set_xlabel('K-uri')
    axes[1].grid(axis='both', linestyle='--', alpha=0.3)

    axes[0].legend(title='Norma', loc='best')
    axes[1].legend(title='Norma', loc='best')

    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    plt.close(fig)
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype='image/png')

@app.route("/preprocessing/<int:training_num>")
def preprocessing_route(training_num):
    preprocessing(training_num)
    return "Preprocesarea a fost finalizata."

@app.route("/image/<int:person>/<int:photo>")
def get_image(person, photo):
    """Serve a face image as base64 encoded PNG."""
    try:
        path = "./att_faces"
        if not os.path.isdir(path):
            return {"error": "Dataset folder not found."}, 404
        
        if person < 1 or person > NR_PERSOANE:
            return {"error": f"Person must be between 1 and {NR_PERSOANE}."}, 400
        
        if photo < 1 or photo > NR_POZE_TOTALE:
            return {"error": f"Photo must be between 1 and {NR_POZE_TOTALE}."}, 400
        
        image_path = f"{path}/s{person}/{photo}.pgm"
        
        if not os.path.isfile(image_path):
            return {"error": "Image file not found."}, 404
        
        # Read the image
        img = cv2.imread(image_path, 0)  # type: ignore
        if img is None:
            return {"error": "Failed to read image."}, 500
        
        # Encode as PNG
        success, buffer = cv2.imencode('.png', img)  # type: ignore
        if not success:
            return {"error": "Failed to encode image."}, 500
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "person": person,
            "photo": photo,
            "image": f"data:image/png;base64,{img_base64}"
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/search/<int:photo_index>/<string:method>/<string:norma>/<int:k>")
def search_route(photo_index, method, norma, k):
    """Search for a matching face and return the matched image."""
    try:
        if not PREPROCESSED:
            return {"error": "Datele nu au fost preprocesate."}, 400
        
        # Perform search
        if method == "nn":
            pozitia = nn(norma, T[:, photo_index])
        elif method == "k_nn":
            if k is None:
                return {"error": "Parametrul k este necesar pentru metoda k_nn."}, 400
            pozitia = k_nn(norma, T[:, photo_index], k)
        else:
            return {"error": "Metoda invalida. Foloseste 'nn' sau 'k_nn'."}, 400
        
        # Calculate matched person and photo from position in training matrix
        matched_person = int((pozitia // NR_POZE_ANTRENARE) + 1)
        matched_photo = int((pozitia % NR_POZE_ANTRENARE) + 1)
        
        # Get the image vector from training matrix
        matched_vector = A[:, pozitia]
        
        # Reshape to image dimensions (112x92 for AT&T faces)
        matched_image = matched_vector.reshape(112, 92)
        
        # Encode as PNG
        success, buffer = cv2.imencode('.png', matched_image.astype(np.uint8))  # type: ignore
        if not success:
            return {"error": "Failed to encode matched image."}, 500
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "matched_person": matched_person,
            "matched_photo": matched_photo,
            "matched_position": int(pozitia),
            "image": f"data:image/png;base64,{img_base64}",
            "method": method,
            "norm": norma,
            "k": k if method == "k_nn" else None
        }
    except Exception as e:
        return {"error": str(e)}, 500