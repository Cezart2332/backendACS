import os
import time
import statistics as st
import cv2
import numpy as np
from numpy import linalg as la

from flask import Flask

poze = []
NR_PERSOANE = 40
NR_POZE_TOTALE = 10
A = None
T = None
NR_POZE_ANTRENARE = 0
PREPROCESSED = False

norme = ["Manhattan", "Euclidean", "Infinit", "Cosinus"]

def preprocessing(training_num):
    global A, T, NR_POZE_ANTRENARE, PREPROCESSED
    NR_POZE_ANTRENARE = training_num
    A = np.zeros([10304, 320])
    T = np.zeros([10304, 80])
    path = "/home/cezar/Desktop/ProiectACS/Lab2/att_faces"
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Folderul '{path}' nu exista. Verifica calea.")

    for i in range(1, NR_PERSOANE + 1):
        person_path = path + "/s" + str(i) + "//"
        for j in range(1, NR_POZE_ANTRENARE + 1):
            person_training_path = person_path + str(j) + ".pgm"
            training_photo = np.array(cv2.imread(person_training_path, 0)) # type: ignore
            training_photo = training_photo.reshape(
                10304,
            )
            A[:, 8 * (i - 1) + (j - 1)] = training_photo

    for i in range(1, NR_PERSOANE + 1):
        person_path = path + "/s" + str(i) + "//"
        for j in range(NR_POZE_ANTRENARE + 1, NR_POZE_TOTALE + 1):
            person_test_path = person_path + str(j) + ".pgm"
            test_photo = np.array(cv2.imread(person_test_path, 0)) # type: ignore
            test_photo = test_photo.reshape(
                10304,
            )
            coloana = 2 * (i - 1) + (j - 9)
            T[:, coloana] = test_photo
    PREPROCESSED = True

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
    pozitia = st.mode(pozitii) * 8
    return pozitia


def statistics():
    if PREPROCESSED is False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing() mai intai.")
    nr_total_teste = NR_PERSOANE * (10 - NR_POZE_ANTRENARE)
    rata_recunoastere_text = "Statistici nn:\n"
    timp_interogare_text = "Statistici nn:\n"
    print("Statistici nn:")
    for norma in norme:
        nr_recunoasteri_corecte = 0
        timp_total_interogare = 0
        for j in range(len(T[0])):
            t0 = time.perf_counter()
            persoana_testata = j // 2
            persoane_cautata = nn(norma, T[:, j]) // 8
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
                persoana_testata = j // 2
                persoane_cautata = k_nn(norma, T[:, j], k) // 8
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
    return rata_recunoastere_text, timp_interogare_text

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/statistics")
def statistics_route():
    rata_recunoastere_text, timp_interogare_text = statistics()
    return f"<pre>{rata_recunoastere_text}\n{timp_interogare_text}</pre>"

@app.route("/preprocessing/<int:training_num>")
def preprocessing_route(training_num):
    preprocessing(training_num)
    return "<p>Preprocessing realizat cu success</p>"
