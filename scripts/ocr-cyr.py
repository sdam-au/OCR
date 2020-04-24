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
resp = conf[0].get(conf[1] + "/SDAM_data/OCR/PDFtexts/Cyrilic")
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
        pages_dict= {}
        for page in doc: ### or you can specify: doc(start, end, step):
            pix = page.getPixmap(matrix = fitz.Matrix(2, 2), colorspace="csGRAY")  # try "csGRAY"
            img = pix2np(pix)
            kernel = np.ones((2, 2), np.uint8)
            img_er = cv2.erode(img, kernel, iterations=1)
            txt = pytesseract.image_to_string(img_er, lang="bul+eng")
            pages_dict["page " + str(i)] = txt 
            i = i+1
        return pages_dict
for filename in filenames: 
    print(datetime.now(), "started to read " + filename)
    resp = conf[0].get(conf[1] + "/SDAM_data/OCR/PDFtexts/Cyrilic/" + filename)
    doc = fitz.open(stream=resp.content, filetype="pdf")
    pages_dict = get_text(doc)
    sddk.write_file("/SDAM_data/OCR/outputs/" + filename.rpartition(".")[0] + ".json", pages_dict,  conf)
    print(datetime.now(), "ended ocr analysis of " + filename + " and saved it to sciencedata.dk")
                                                                         [ Read 60 lines ]
