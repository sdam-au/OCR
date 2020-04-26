### binding for tesseract:
import pytesseract
### computer vision:
import cv2
### computer vision relies to a substantial extent on numpy arrays
import numpy as np
### PyMuPDF is called fitz:
import fitz
### to plot pages and everything else:
from matplotlib import pyplot as plt
### to import data from sciencedata.dk
import sddk
from bs4 import BeautifulSoup
from datetime import datetime
### configure sddk session
conf = sddk.configure_session_and_url("SDAM_root", "648597@au.dk")

directory = input("you are in " + conf[1] + ", specify subdirectory: ")
language = input("specify language (use '+' for more languages): ")

resp = conf[0].get(conf[1] + directory)
soup = BeautifulSoup(resp.content)
soup
filenames = []
for a in soup.find_all("a"):
    a_str = str(a.get_text())
    if ".pdf" in a_str:
        filenames.append(a_str)
print("files in the folder: ")
print(filenames)
def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    #im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im
def get_text(doc):
        i = 1
        pages = ""
        for page in doc: ### or you can specify: doc(start, end, step):
            pix = page.getPixmap(matrix = fitz.Matrix(2, 2), colorspace="csGRAY")  # try "csGRAY"
            img = pix2np(pix)
            kernel = np.ones((2, 2), np.uint8)
            img_er = cv2.erode(img, kernel, iterations=1)
            txt = pytesseract.image_to_string(img_er, lang=language) + "\n\n[end-of-page" + str(i) + "]\n\n"
            pages += txt 
            i = i+1
        return pages

for filename in filenames: 
    print(datetime.now(), "started to read " + filename)
    resp = conf[0].get(conf[1] + directory + filename)
    doc = fitz.open(stream=resp.content, filetype="pdf")
    pages_str = get_text(doc)
    filepathname = "/SDAM_data/OCR/outputs/" + filename.rpartition(".")[0] + ".txt"
    conf[0].put(conf[1] + filepathname, data=pages_str.encode('utf-8'))
    print(datetime.now(), "ended ocr analysis of " + filename + " and saved it to sciencedata.dk")
