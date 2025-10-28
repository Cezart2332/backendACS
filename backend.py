import os
import time
import statistics as st
import cv2
import numpy as np
from numpy import linalg as la

from flask import Flask, send_file, make_response
from io import StringIO, BytesIO
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

norme = ["Manhattan", "Euclidean", "Infinit", "Cosinus"]

def preprocessing(training_num):
    global A, T, NR_POZE_ANTRENARE, NR_POZE_TESTARE, PREPROCESSED, STATS_RUNNED
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

def nn(norm, p):
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing() mai intai.")
    z = np.zeros(len(A[0]))
    for i in range(len(A[0])):
        if norm == "Manhattan":
            z[i] = la.norm(A[:, i] - p, 1)
        elif norm == "Euclidean":
            z[i] = la.norm(A[:, i] - p, 2)
        elif norm == "Infinit":
            z[i] = la.norm(A[:, i] - p, np.inf)
        elif norm == "Cosinus":
            z[i] = 1 - (np.dot(A[:, i], p)) / (la.norm(A[:, i], 2) * la.norm(p, 2))
        else:
            raise ValueError("Norma Invalida!")
    pozitia = np.argmin(z)
    return pozitia


def k_nn(norm, p, k):
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing() mai intai.")
    z = np.zeros(len(A[0]))
    for i in range(len(A[0])):
        if norm == "Manhattan":
            z[i] = la.norm(A[:, i] - p, 1)
        elif norm == "Euclidean":
            z[i] = la.norm(A[:, i] - p, 2)
        elif norm == "Infinit":
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
    global STATS_RUNNED, rata_recunoastere_text, timp_interogare_text
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing() mai intai.")
    if STATS_RUNNED is True:
        raise Exception("Statistici deja calculate. Ruleaza preprocessing() cu alta configuratie pentru a reseta.")
    nr_total_teste = NR_PERSOANE * (10 - NR_POZE_ANTRENARE)
    rata_recunoastere_text = "Statistici nn:\n"
    timp_interogare_text = "Statistici nn:\n"
    print("Statistici nn:")
    for norma in norme:
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
        print(f"Rata de recunoastere norma={norma}: {rr*100:.2f}%")
        tmi = timp_total_interogare / nr_total_teste
        print(f"Timp mediu de interogare norma={norma}: {tmi:.8f}")
        rata_recunoastere_text += f"Rata de recunoastere norma={norma}: {rr*100:.2f}%\n"
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
            print(f"Rata de recunoastere norma={norma} si k={k}: {rr*100:.2f}%")
            tmi = timp_total_interogare / nr_total_teste
            print(f"Timp mediu de interogare norma={norma} si k={k}: {tmi:.8f}")
            rata_recunoastere_text += (
                f"Rata de recunoastere norma={norma} si k={k}: {rr*100:.8f}%\n"
            )
            timp_interogare_text += (
                f"Timp mediu de interogare norma={norma} si k={k}: {tmi:.8f}\n"
            )
    STATS_RUNNED = True

app = Flask(__name__)

# Enable CORS for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/statistics")
def statistics_route():
    if STATS_RUNNED is False:
        statistics()
    return f"<pre>{rata_recunoastere_text}\n{timp_interogare_text}</pre>"

@app.route("/statistics/export/<format_type>")
def export_statistics(format_type):
    """Export statistics as CSV or TXT file."""
    if PREPROCESSED is False:
        return {"error": "Datele nu au fost preprocesate."}, 400
    if STATS_RUNNED is False:
        statistics()
    
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
        # Parse the statistics and create CSV
        output = StringIO()
        writer = csv.writer(output)
        
        # Write CSV headers
        writer.writerow(['Algoritm', 'Norma', 'K', 'Rata Recunoastere (%)', 'Timp Mediu Interogare (s)'])
        
        # Parse NN statistics
        lines = rata_recunoastere_text.split('\n')
        time_lines = timp_interogare_text.split('\n')
        
        for i, line in enumerate(lines):
            if 'Rata de recunoastere norma=' in line:
                # Extract norm and rate
                norm = line.split('norma=')[1].split(':')[0]
                rate = line.split(': ')[1].replace('%', '').strip()
                
                # Find corresponding time
                time_val = ''
                for time_line in time_lines:
                    if f'norma={norm}' in time_line and 'k=' not in time_line:
                        time_val = time_line.split(': ')[1].strip()
                        break
                
                if ' si k=' in line:
                    # k-NN entry
                    k_val = line.split('k=')[1].split(':')[0]
                    writer.writerow(['k-NN', norm, k_val, rate, time_val])
                else:
                    # NN entry
                    writer.writerow(['NN', norm, '-', rate, time_val])
        
        # Convert to bytes
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

@app.route("/preprocessing/<int:training_num>")
def preprocessing_route(training_num):
    preprocessing(training_num)
    return "Preprocesarea a fost finalizata."
