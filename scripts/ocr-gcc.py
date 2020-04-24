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
### configure sddk session
conf = sddk.configure_session_and_url("SDAM_root", "648597@au.dk")
resp = conf[0].get(conf[1] + "/SDAM_data/OCR/PDFtexts/Cyrilic/AOR'1990_1989.pdf")
doc = fitz.open(stream=resp.content, filetype="pdf")
def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    #im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im
i = 1
pages_txt= {}
for page in doc: ### or you can specify: doc(start, end, step):
    pix = page.getPixmap(matrix = fitz.Matrix(2, 2), colorspace="csGRAY")  # try "csGRAY"
    img = pix2np(pix)
    kernel = np.ones((2, 2), np.uint8)
    img_er = cv2.erode(img, kernel, iterations=1)
    txt = pytesseract.image_to_string(img_er, lang="bul+eng")
    pages_txt["page " + str(i)] = txt 
    i = i+1
sddk.write_file("/SDAM_data/OCR/outputs/" + ¨)
