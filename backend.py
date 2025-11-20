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
RC = None
HQPB_RC = None
MEDIA_RC = None
PROIECTII_RC = None
NR_PERSOANE = 40
NR_POZE_TOTALE = 10
A = None
T = None
NR_POZE_ANTRENARE = 0
NR_POZE_TESTARE = 0
PREPROCESSED = False
STATS_RUNNED = False
STATS_DATA = None
MEDIA = None
HQPB = None
K = 100
PROIECTII = None
Q = None
ALFA = None
BETA = None
PROIECTII_LANCZOS = None
PREPROCESSING_TIMES = {}

PREPROCESSING_LABELS = [
    ("Eigenfaces", "Eigenfaces"),
    ("Eigenfaces with RC", "Eigenfaces with RC"),
    ("Lanczos", "Lanczos"),
]

norme = ["Manhattan", "Euclidean", "Infinity", "Cosinus"]

def build_rc():
    global RC 
    RC = np.zeros((10304, NR_PERSOANE), dtype=float)
    for i in range(NR_PERSOANE):
        start = i * NR_POZE_ANTRENARE
        end = start + NR_POZE_ANTRENARE
        Ai = A[:, start:end]

        RC[:, i] = np.mean(Ai, axis=1)
    

def preprocceseg():
    global MEDIA,HQPB,PROIECTII, PREPROCESSING_TIMES
    start_time = time.perf_counter()
    B = A.copy()
    MEDIA = np.mean(A, axis=1)
    B = (B.T - MEDIA).T
    l = np.dot(B.T,B)
    d,v = la.eig(l)
    v = np.dot(B,v)
    indici = np.argsort(d)[::-1]
    v = v[:,indici]
    k = K
    HQPB = v[:, :k]
    PROIECTII = np.dot(B.T, HQPB)
    PREPROCESSING_TIMES["Eigenfaces"] = time.perf_counter() - start_time
def preprocesseg_rc():
    build_rc()
    global MEDIA_RC, PROIECTII_RC, PREPROCESSING_TIMES
    start_time = time.perf_counter()

    RC_COPY = RC.copy()
    MEDIA_RC = np.mean(RC_COPY, axis=1)
    RC_COPY = (RC_COPY.T - MEDIA).T
    PROIECTII_RC = np.dot(RC_COPY.T, HQPB)
    PREPROCESSING_TIMES["Eigenfaces with RC"] = time.perf_counter() - start_time

def preproccesslanczos(K):
    global A, Q, ALFA, BETA, PROIECTII_LANCZOS, PREPROCESSING_TIMES
    start_time = time.perf_counter()
    
    m = A.shape[0]             # nr. de pixeli
    # q0..q_{K+1}  => K+2 coloane
    q = np.zeros((m, K+2))
    alpha = np.zeros(K+1)
    beta = np.zeros(K+2)

    # q0 = 0 (deja e)
    # q1 = [1,1,...,1] normalizat
    q[:, 1] = np.ones(m)
    q[:, 1] /= la.norm(q[:, 1])

    beta[1] = 0.0   # β1 = 0

    # i = 1..K
    for i in range(1, K+1):
        qi = q[:, i]

        # ωi = A(A^T qi) − βi q_{i−1}
        omega = A @ (A.T @ qi) - beta[i] * q[:, i-1]

        # αi = <ωi, qi>
        alpha[i] = np.dot(omega, qi)

        # ωi = ωi − αi qi
        omega = omega - alpha[i] * qi

        # β_{i+1} = ||ωi||
        beta[i+1] = la.norm(omega)
        if beta[i+1] < 1e-12:
            break

        # q_{i+1} = ωi / β_{i+1}
        q[:, i+1] = omega / beta[i+1]

    ALFA = alpha
    BETA = beta

    # HQPB = q fără primele 2 coloane
    Q = q[:, 2:]          

    # proiecții exact ca la Eigenfaces
    B = A.copy()
    MEDIA = np.mean(A, axis=1)
    B = (B.T - MEDIA).T
    PROIECTII_LANCZOS = B.T @ Q

    PREPROCESSING_TIMES["Lanczos"] = time.perf_counter() - start_time

    return PROIECTII_LANCZOS



def preprocessing(training_num):
    global A, T, NR_POZE_ANTRENARE, NR_POZE_TESTARE, PREPROCESSED, STATS_RUNNED, STATS_DATA,MEDIA,HQPB,PROIECTII, PREPROCESSING_TIMES
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
    PREPROCESSING_TIMES = {}
    preprocceseg()
    preprocesseg_rc()
    preproccesslanczos(100)  # Initialize Lanczos with K=100 components
    PREPROCESSED = True
    STATS_RUNNED = False
    STATS_DATA = None



def nn(norm, p,A=None):
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing mai intai.")
    if A is None:
        A = globals()['A']
    zeros = len(A[0])
    z = np.zeros(zeros)
    for i in range(zeros):
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
    
def eigenfaces(norm,p):
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing mai intai.")
    p = p - MEDIA
    p_test = np.dot(p, HQPB)
    pozitia = nn(norm, p_test, PROIECTII.T)
    return pozitia
def eigenfaces_rc(norm, p):
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing mai intai.")

    centered = p - MEDIA
    p_test = np.dot(centered, HQPB)

    pozitia = nn(norm, p_test, PROIECTII_RC.T) 

    return int(pozitia * NR_POZE_ANTRENARE)

def lanczos(norm, p):
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing mai intai.")
    p = p - MEDIA
    p_test = np.dot(p, Q)
    pozitia = nn(norm, p_test, PROIECTII_LANCZOS.T)
    return pozitia

def statistics():
    global STATS_RUNNED, rata_recunoastere_text, timp_interogare_text, STATS_DATA
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing mai intai.")
    if STATS_RUNNED is True:
        raise Exception("Statistici deja calculate. Ruleaza preprocessing() cu alta configuratie pentru a reseta.")
    nr_total_teste = NR_PERSOANE * (10 - NR_POZE_ANTRENARE)
    STATS_DATA = {"nn": [], "k_nn": [], "eigenfaces": [], "eigenfaces_rc": [], "lanczos": []}
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
    print("Statistici eigenfaces:")
    for norma in norme:
        print(f"Calcul statistici pentru norma: {norma}")
        nr_recunoasteri_corecte = 0
        timp_total_interogare = 0
        for j in range(len(T[0])):
            t0 = time.perf_counter()
            persoana_testata = j // NR_POZE_TESTARE
            persoane_cautata = eigenfaces(norma, T[:, j]) // NR_POZE_ANTRENARE
            t1 = time.perf_counter()
            timp_total_interogare += t1 - t0
            if persoana_testata == persoane_cautata:
                nr_recunoasteri_corecte = nr_recunoasteri_corecte + 1
        rr = nr_recunoasteri_corecte / nr_total_teste
        rata_procente = rr * 100
        tmi = timp_total_interogare / nr_total_teste
        STATS_DATA["eigenfaces"].append({"norm": norma, "rate": rata_procente, "time": tmi})
        print(f"Rata de recunoastere norma={norma}: {rata_procente:.2f}%")
        print(f"Timp mediu de interogare norma={norma}: {tmi:.8f}")
        rata_recunoastere_text += f"Rata de recunoastere norma={norma}: {rata_procente:.2f}%\n"
        timp_interogare_text += f"Timp mediu de interogare norma={norma}: {tmi:.8f}\n"
    print("Statistici Eigenfaces with RC:")
    for norma in norme:
        print(f"Calcul statistici pentru norma: {norma}")
        nr_recunoasteri_corecte = 0
        timp_total_interogare = 0
        for j in range(len(T[0])):
            t0 = time.perf_counter()
            persoana_testata = j // NR_POZE_TESTARE
            persoane_cautata = eigenfaces_rc(norma, T[:, j]) // NR_POZE_ANTRENARE
            t1 = time.perf_counter()
            timp_total_interogare += t1 - t0
            if persoana_testata == persoane_cautata:
                nr_recunoasteri_corecte = nr_recunoasteri_corecte + 1
        rr = nr_recunoasteri_corecte / nr_total_teste
        rata_procente = rr * 100
        tmi = timp_total_interogare / nr_total_teste
        STATS_DATA["eigenfaces_rc"].append({"norm": norma, "rate": rata_procente, "time": tmi})
        print(f"Rata de recunoastere norma={norma}: {rata_procente:.2f}%")
        print(f"Timp mediu de interogare norma={norma}: {tmi:.8f}")
        rata_recunoastere_text += f"Rata de recunoastere norma={norma}: {rata_procente:.2f}%\n"
        timp_interogare_text += f"Timp mediu de interogare norma={norma}: {tmi:.8f}\n"
    
    print("Statistici Lanczos:")
    for norma in norme:
        print(f"Calcul statistici pentru norma: {norma}")
        nr_recunoasteri_corecte = 0
        timp_total_interogare = 0
        for j in range(len(T[0])):
            t0 = time.perf_counter()
            persoana_testata = j // NR_POZE_TESTARE
            persoane_cautata = lanczos(norma, T[:, j]) // NR_POZE_ANTRENARE
            t1 = time.perf_counter()
            timp_total_interogare += t1 - t0
            if persoana_testata == persoane_cautata:
                nr_recunoasteri_corecte = nr_recunoasteri_corecte + 1
        rr = nr_recunoasteri_corecte / nr_total_teste
        rata_procente = rr * 100
        tmi = timp_total_interogare / nr_total_teste
        STATS_DATA["lanczos"].append({"norm": norma, "rate": rata_procente, "time": tmi})
        print(f"Rata de recunoastere norma={norma}: {rata_procente:.2f}%")
        print(f"Timp mediu de interogare norma={norma}: {tmi:.8f}")
        rata_recunoastere_text += f"Rata de recunoastere norma={norma}: {rata_procente:.2f}%\n"
        timp_interogare_text += f"Timp mediu de interogare norma={norma}: {tmi:.8f}\n"

        
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
        output.write("=== TIMPI PREPROCESARE (s) ===\n")
        if PREPROCESSING_TIMES:
            for label, key in PREPROCESSING_LABELS:
                durata = PREPROCESSING_TIMES.get(key)
                if durata is None:
                    output.write(f"{label}: indisponibil\n")
                else:
                    output.write(f"{label}: {durata:.6f}\n")
        else:
            output.write("Date indisponibile. Ruleaza preprocessing.\n")
        output.write("\n")
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

        writer.writerow(['Preprocessing Algorithm', 'Duration (s)'])
        if PREPROCESSING_TIMES:
            for label, key in PREPROCESSING_LABELS:
                durata = PREPROCESSING_TIMES.get(key)
                writer.writerow([label, f"{durata:.6f}" if durata is not None else 'N/A'])
        else:
            writer.writerow(['Data unavailable', 'Run preprocessing first'])

        writer.writerow([])

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
        for entry in STATS_DATA["eigenfaces"]:
            writer.writerow([
                'Eigenfaces',
                entry['norm'],
                '-',
                f"{entry['rate']:.2f}",
                f"{entry['time']:.8f}"
            ])
        for entry in STATS_DATA["eigenfaces_rc"]:
            writer.writerow([
                'Eigenfaces with RC',
                entry['norm'],
                '-',
                f"{entry['rate']:.2f}",
                f"{entry['time']:.8f}"
            ])
        for entry in STATS_DATA["lanczos"]:
            writer.writerow([
                'Lanczos',
                entry['norm'],
                '-',
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

    norms_order = norme

    def build_series(label, entries, extra_filter=None):
        data_by_norm = {}
        for entry in entries:
            key = entry['norm']
            if extra_filter and not extra_filter(entry):
                continue
            data_by_norm.setdefault(key, entry)
        rates = [data_by_norm.get(norm, {}).get('rate', np.nan) for norm in norms_order]
        times = [data_by_norm.get(norm, {}).get('time', np.nan) for norm in norms_order]
        return rates, times

    series = []

    series.append(("NN",) + build_series("NN", STATS_DATA.get("nn", [])))
    series.append(("Eigenfaces",) + build_series("Eigenfaces", STATS_DATA.get("eigenfaces", [])))
    series.append(("Eigenfaces with RC",) + build_series("Eigenfaces with RC", STATS_DATA.get("eigenfaces_rc", [])))
    series.append(("Lanczos",) + build_series("Lanczos", STATS_DATA.get("lanczos", [])))

    knn_entries = STATS_DATA.get("k_nn", [])
    k_values = sorted({entry['k'] for entry in knn_entries})
    for k in k_values:
        label = f"k-NN (k={k})"
        rates, times = build_series(label, knn_entries, extra_filter=lambda e, current_k=k: e['k'] == current_k)
        series.append((label, rates, times))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout='constrained')

    x = np.arange(len(norms_order))
    color_map = plt.cm.get_cmap('tab20', len(series)) if series else None

    for idx, (label, rates, times) in enumerate(series):
        color = color_map(idx) if color_map else None
        axes[0].plot(x, rates, marker='o', label=label, color=color)
        axes[1].plot(x, times, marker='o', label=label, color=color)

    x_labels = norms_order
    axes[0].set_title('Rata de recunoastere (%)')
    axes[0].set_ylabel('Procente (%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, rotation=30, ha='right')
    axes[0].set_xlabel('Norma')
    axes[0].grid(axis='both', linestyle='--', alpha=0.3)

    axes[1].set_title('Timp mediu de interogare (s)')
    axes[1].set_ylabel('Secunde')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=30, ha='right')
    axes[1].set_xlabel('Norma')
    axes[1].grid(axis='both', linestyle='--', alpha=0.3)

    axes[0].legend(title='Algoritm', bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[1].legend(title='Algoritm', bbox_to_anchor=(1.02, 1), loc='upper left')

    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    plt.close(fig)
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype='image/png')


@app.route("/statistics/preprocessing-graph")
def preprocessing_graph():
    if PREPROCESSED is False:
        return {"error": "Datele nu au fost preprocesate."}, 400
    if not PREPROCESSING_TIMES:
        return {"error": "Timpi de preprocesare indisponibili. Ruleaza preprocessing."}, 400

    labels = [label for label, _ in PREPROCESSING_LABELS]
    durations = [PREPROCESSING_TIMES.get(key) for _, key in PREPROCESSING_LABELS]

    if not any(duration is not None for duration in durations):
        return {"error": "Timpi de preprocesare indisponibili."}, 500

    plotted = [duration if duration is not None else 0.0 for duration in durations]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, plotted, color="#4c72b0")

    for bar, duration in zip(bars, durations):
        if duration is None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "N/A", ha='center', va='bottom', color='gray')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{duration:.4f}s", ha='center', va='bottom')

    ax.set_ylabel('Secunde')
    ax.set_title('Comparatie timpi de preprocesare')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

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
        elif method == "Eigenfaces":
            pozitia = eigenfaces(norma,T[:, photo_index])
        elif method == "Eigenfaces with RC":
            pozitia = eigenfaces_rc(norma,T[:, photo_index])
        elif method == "Lanczos":
            pozitia = lanczos(norma, T[:, photo_index])
        else:
            return {"error": "Metoda invalida. Foloseste 'nn', 'k_nn', 'Eigenfaces', 'Eigenfaces with RC', sau 'Lanczos'."}, 400
        
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