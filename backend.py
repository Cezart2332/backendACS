import cv2
import os
import numpy as np
from numpy import linalg as la
import statistics as st
import time

from flask import Flask


poze = []
nrPersoane = 40
nrPozeTotale = 10
A = None  
T = None
nrPozeAntrenare = 0
preprocessed = False

norme = ["Manhattan", "Euclidean", "Infinit", "Cosinus"]


def preprocessing(trainingNum):
    global A, T, nrPozeAntrenare,preprocessed
    nrPozeAntrenare = trainingNum
    A = np.zeros([10304, 320])
    T = np.zeros([10304, 80])
    path = "/home/cezar/Desktop/ProiectACS/Lab2/att_faces"
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Folderul '{path}' nu exista. Verifica calea.")

    for i in range(1, nrPersoane + 1):
        personPath = path + "/s" + str(i) + "//"
        for j in range(1, nrPozeAntrenare + 1):
            personTrainingPath = personPath + str(j) + ".pgm"
            trainingPhoto = np.array(cv2.imread(personTrainingPath, 0)) 
            trainingPhoto = trainingPhoto.reshape(
                10304,
            )
            A[:, 8 * (i - 1) + (j - 1)] = trainingPhoto

    for i in range(1, nrPersoane + 1):
        personPath = path + "/s" + str(i) + "//"
        for j in range(nrPozeAntrenare + 1, nrPozeTotale + 1):
            personTestPath = personPath + str(j) + ".pgm"
            testPhoto = np.array(cv2.imread(personTestPath, 0))
            testPhoto = testPhoto.reshape(
                10304,
            )
            coloana = 2 * (i - 1) + (j - 9)
            T[:, coloana] = testPhoto
            
    preprocessed = True


def NN(norm, A, p):
    if preprocessed == False:
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
            ValueError("Norma Invalida!")
    pozitia = np.argmin(z)
    return pozitia


def kNN(norm, A, p, k):
    if preprocessed == False:   
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
            ValueError("Norma Invalida!")
    indicii = np.argsort(z)[:k]
    pozitii = indicii // nrPozeAntrenare
    pozitia = st.mode(pozitii) * 8
    return pozitia


def statistics(nrPozeAntrenare):
    if preprocessed == False:
        raise Exception("Datele nu au fost preprocesate. Ruleaza preprocessing() mai intai.")
    nrTotalTeste = nrPersoane * (10 - nrPozeAntrenare)
    rataRecunoastereText = "Statistici NN:\n"
    timpInterogareText = "Statistici NN:\n"
    print("Statistici NN:")
    for norma in norme:
        nrRecunoasteriCorecte = 0
        timpTotalInterogare = 0
        for j in range(len(T[0])):
            t0 = time.perf_counter()
            persoanaTestata = j // 2
            persoanaCautata = NN(norma, A, T[:, j]) // 8
            t1 = time.perf_counter()
            timpTotalInterogare += t1 - t0
            if persoanaTestata == persoanaCautata:
                nrRecunoasteriCorecte = nrRecunoasteriCorecte + 1
        rr = nrRecunoasteriCorecte / nrTotalTeste
        print(f"Rata de recunoastere norma={norma}: {rr*100:.2f}%")
        tmi = timpTotalInterogare / nrTotalTeste
        print(f"Timp mediu de interogare norma={norma}: {tmi:.8f}")
        rataRecunoastereText += f"Rata de recunoastere norma={norma}: {rr*100:.2f}%\n"
        timpInterogareText += f"Timp mediu de interogare norma={norma}: {tmi:.8f}\n"

    rataRecunoastereText += "Statistici kNN:\n"
    timpInterogareText += "Statistici kNN:\n"
    print("Statistici kNN:")
    for k in range(1, 9, 2):
        for norma in norme:
            nrRecunoasteriCorecte = 0
            timpTotalInterogare = 0
            for j in range(len(T[0])):
                t0 = time.perf_counter()
                persoanaTestata = j // 2
                persoanaCautata = kNN(norma, A, T[:, j], k) // 8
                t1 = time.perf_counter()
                timpTotalInterogare += t1 - t0
                if persoanaTestata == persoanaCautata:
                    nrRecunoasteriCorecte = nrRecunoasteriCorecte + 1
            rr = nrRecunoasteriCorecte / nrTotalTeste
            print(f"Rata de recunoastere norma={norma} si k={k}: {rr*100:.2f}%")
            tmi = timpTotalInterogare / nrTotalTeste
            print(f"Timp mediu de interogare norma={norma} si k={k}: {tmi:.8f}")
            rataRecunoastereText += (
                f"Rata de recunoastere norma={norma} si k={k}: {rr*100:.8f}%\n"
            )
            timpInterogareText += (
                f"Timp mediu de interogare norma={norma} si k={k}: {tmi:.8f}\n"
            )
    return rataRecunoastereText, timpInterogareText


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/statistics")
def statistics_route():
    rataRecunoastereText, timpInterogareText = statistics(nrPozeAntrenare)
    return f"<pre>{rataRecunoastereText}\n{timpInterogareText}</pre>"


@app.route("/preprocessing/<int:trainingNum>")
def preprocessing_route(trainingNum):
    preprocessing(trainingNum)
    return f"<p>Preprocessing realizat cu {trainingNum} poze de antrenare. si {nrPozeTotale - trainingNum} poze de testare.</p>"
