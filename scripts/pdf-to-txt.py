# binding for tesseract:
import pytesseract
# computer vision:
import cv2
# computer vision relies to a substantial extent on numpy arrays
import numpy as np
# PyMuPDF is called fitz:
import fitz
# to plot pages and everything else:
from datetime import datetime
# configure sddk session


# CRUCIAL FUNCTIONS
def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    #im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im

def get_text(doc):
    i = 1
    pages = ""
    for page in doc: # or you can specify: doc(start, end, step):
        pix = page.getPixmap(matrix = fitz.Matrix(2, 2), colorspace="csGRAY")  # try "csGRAY"
        img = pix2np(pix)
        kernel = np.ones((2, 2), np.uint8)
        img_er = cv2.erode(img, kernel, iterations=1)
        txt = pytesseract.image_to_string(img_er, lang=language) + "\n\n[end-of-page" + str(i) + "]\n\n"
        pages += txt 
        i = i+1
    return pages

inputfile = input("file for ocr: ") # (works interactively)
try:
    doc = fitz.open(inputfile)
except:
    print("reading of the file failed, have you correctly specified its relative path from here? Try again:")
    inputfile = input("file for ocr: ")
    doc = fitz.open(inputfile)

language = input("specify language of the pdf (use '+' for more languages): ")

print(datetime.now(), "ocr analysis started")
pages_str = get_text(doc)

outputfile = input("specify name of the output file: ")

# SAVE THE FILE
file = open(outputfile, "w")
file.write(pages_str)
print(datetime.now(), "ocr analysis ended and the output file was saved.")