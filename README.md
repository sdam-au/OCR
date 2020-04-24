# OCR_tests

## Purpose

The purpose of this repo and this document is to report on my attempt to do an optical character recognition (OCR) of archaeological reports using python. It is a challenging task, since some of the source pdfs are in Cyrillic and do not have a very good quality.

## Authors

* Vojtěch Kaše, Aarhus University/University of West Bohemia

## Licence

CC-BY-SA 4.0, see attached License.md

## Data
The data for this project are  scanned PDFs of  various archaeological reports, differing in language and quality.

## Prerequisites 

### Software

The tools in this repo have been tested on Mac with a local installation of Python 3. While I normally work online in Google Colab, here I am working with my local Python, since it relies on the other two programs through bindings.

1. Python 3
2. Tesseract
3. MuPDF

## REPORT

### Tesseract

**Description**: Tesseract is an optical character recognition engine for various operating systems. It is free software, released under the Apache License, Version 2.0, and development has been sponsored by Google since 2006. In 2006, Tesseract was considered one of the most accurate open-source OCR engines then available. ([wiki](http://tesseract))
    

* [official tesseract repo](https://github.com/tesseract-ocr/tesseract)
* [repo with simple python example](https://github.com/MauryaRitesh/OCR-Python) - The aim of this Repository is to be able to recognise text from an image file using the Tesseract Library in the Python Programming Language ([video tutorial with simple .ipynb](https://www.youtube.com/watch?v=fn7A50rBtD0&feature=youtu.be)).
*  [extensive video tutorial](https://www.youtube.com/watch?v=kxHp5ng6Rgw)
   
#### Installation:
**a) linux**:
* `sudo apt-get install tesseract-ocr`

**b) macOSx**:  you need either `brew` or `macports` installed, ports are easier, but you must have “Command Line Tools for Xcode”, then you can run:
* `$ sudo port install tesseract` or:
* `$ sudo port install tesseract-<language>` to install specific language version, e.g. `tesseract-ces` for Czech
* alternatively, with brew you run `brew install tesseract`, howerver, then I do not know what is a syntax for installing individual language mutations, because `brew install tesseract-lang` does not work 
* overview of languages is [here](https://www.macports.org/ports.php?by=name&substr=tesseract-) 
* you might face problems with permissions to write into certain directories, to solve this. run:
	* `shell sudo chown -R au648597 /usr/local/lib/pkgconfig /usr/local/share/info /usr/local/share/man/man3`
	* `chmod u+w /usr/local/lib/pkgconfig /usr/local/share/info /usr/local/share/man/man3`

**Simple command line example:** 
- move into a folder close to some test data and run
`tesseract data/test-image.png data/output.txt`
  
### PyTesseract
**Simple pytesseract example from console**
`pytesseract` is a python wrapper for tesseract. To open images inside python, you also need `pillow` package, therefore:
1) install both dependencies using pip (or `pip3` for python 3 specifically):
`$ pip3 install pytesseract pillow`
 2) open python
 `$ python3`
  3) import the packages: 
```python 
>>> from PIL import Image
>>> import pytesseract
```
4) define a variable `string` and assign to it an output of the `image_to_string()` function:
```python
string = pytesseract.image_to_string(Image.open(“test-image.png”), lang=”eng”)
```
 5) print the output:
```python
>>> print(string)
```
It works very well with Czech texts as well. There are many apps relying on tesseract, like [Ancient Greek OCR app](https://ancientgreekocr.org). Always you have firstly install the language by running:
`$ sudo port install tesseract-grc`

### Convert PDF to text in python
(based on [this](https://www.youtube.com/watch?v=pf7OONW7l54) tutorial.)

```bash
$ pip3 install wand
```
`wand` is a `ctypes`-based simple ImageMagick binding for Python - [docs](http://docs.wand-py.org).
 To work with wand, you must have ImageMagick installed as well:
 ```bash
 $ brew install freetype imagemagick
 ```

A complete python script for straightforward production of texts:

```python
### REQUIREMENTS
import io # navigating files
from PIL import Image 
import pytesseract
from wand.image import Image as wi # working with PDFs and images

### READ PDF FILE AS JPEG
pdf = wi(filename = input('input filename: '), resolution = 300)
pdfImg = pdf.convert('jpeg')
### TRANSFORM IT TO IMAGES 
imgBlobs = []
for img in pdfImg.sequence:
	page = wi(image = img)
	imgBlobs.append(page.make_blob('jpeg'))

### EXTRACT TEXT FROM THE IMAGES
language = input("language code (e.g. 'eng':" )
extracted_text = []
i = 1
for imgBlob in imgBlobs:
	im = Image.open(io.BytesIO(imgBlob))
	text = pytesseract.image_to_string(im, lang = language)
	extracted_text.append(text + " [end-of-page" + str(i) + "]")
	i += 1

### SAVE THE FILE
file = open(input("output filename: "),"w")
file.write(" ".join(extracted_text))
```
### Testing Tesseract with Cyrilic

First, you have have to install the language:
```bash
$ sudo port install tesseract-bul
```
Second, you can run the same script file as above, i.e. `pdf3text.py`. As the script proceeds, it asks you to specify three things:
* path and name of the input pdf file
* language of the pdf
* path and name for the output file

In the example above, I was reading pdf files using `Image()` function from `wand.image` module.
```python
pdf = wi(filename = "data/test-cyr_p4.pdf", resolution=300)
```
I was experimenting with resolution, comparing 100, 200, 300, and 600, but have not seen any difference. 

Working on this I actually realize that I want to try a different tool for working with pdfs. This brought me to PyMuPDF.


### MuPdf & PyMuPdf
The main motivation was to make straightforward move from pdfs to computer vision readable objects used by `cv2` library, what would enable me to do some additional transformations. The original script above, using wand, was perhaps not the best solution for this purpose (as a binding of ImageMagick, the python package is not so much documented). Therefore I turned to PyMuPdf, based on MuPdf (following [this](https://stackoverflow.com/questions/53059007/python-opencv) thread).
First, you have to install MuPDF [docs](https://mupdf.com/docs/index.html).  From bash, you can install it using `brew`. In my case I first had to install xquartz however:

You can either install it straightforwardly:
```bash
$ pip3 install pymupdf
```
Or you can firstly install MuPDF as I did:
```bash
$ brew cask install xquartz
```
and then, I was finally able to run:

```bash
$ brew install mupdf
```
I got it here: `/usr/local/Cellar/mupdf`

### PyMuPDF
(All the code below is from jupyter notebook `scripts/pdf-and-image-preprocessing.ipynb`).

PyMuPDF has a very intuitive usage (see the [tutorial](https://pymupdf.readthedocs.io/en/latest/tutorial/)). To read a pdf, you run just:

```python
doc = fitz.open("data/test-cyr.pdf") ### open the pdf
```
Then you can easily iterate over pages and do anything you wish (look for annotations, links, etcs.). 

The most important thing for us is to render images into a `Pixmap` object. Pixmap object is a RGB image of a page. As the documentation says, "[m]ethod [`Page.getPixmap()`](https://pymupdf.readthedocs.io/en/latest/page/#Page.getPixmap "Page.getPixmap") offers lots of variations for controlling the image: resolution, colorspace (e.g. to produce a grayscale image or an image with a subtractive color scheme), transparency, rotation, mirroring, shifting, shearing, etc."

```python
for page in doc(start, end, step):
	pix = page.getPixmap(colorspace = "GRAY") # try "csGRAY"
```

There are very nice [recipes](https://pymupdf.readthedocs.io/en/latest/faq/) for this procedure. For instance, to get a better resolution, you can use matrix parameter and so to say to "zoom in".

To really test different parametrizations of getPicxmap, I produced 5 images of the same area from a third page of a pdf `doc`.  I also defined a function called `rect` to capture a rectangle area of my interest defined by a list of ratio values: `[start height, end height, start width, end width]`.
```python
def rect(img,rect):
    '''return rectangle defined by side ratio'''
    h = img.shape[0]
    w = img.shape[1]
    return img[int(h * rect[0]):int((h * rect[1])), int(w * rect[2]):int((w * rect[3]))]

test_img = doc[2] ### select the page
test_imgs = [] 
test_imgs.append(pix2np(test_img.getPixmap()))
test_imgs.append(pix2np(test_img.getPixmap(colorspace="csGRAY")))
test_imgs.append(pix2np(test_img.getPixmap(matrix = fitz.Matrix(2, 2))))
test_imgs.append(pix2np(test_img.getPixmap(matrix = fitz.Matrix(2, 2), colorspace="csGRAY")))
test_imgs.append(pix2np(test_img.getPixmap(matrix = fitz.Matrix(3, 3), colorspace="csGRAY")))

test_imgs = [rect(img, [0.07, 0.57, 0.5, 1]) for img in test_imgs] # defined the area
```
Already a very preliminary look at these output indicates that **zooming** actually means a lot of improvement. A zoom with matrix = (2,2) appears to produce the best results.

### Morphiological transformations with CV2.

Subsequently, I tested some basic [morphological transformations](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html), namely **Erosion** , **Dilation**, and **Closing**.

So I have the image extracted from the pdf with these parameters:
```python
img = test_imgs.append(pix2np(test_img.getPixmap(matrix = fitz.Matrix(3, 3), colorspace="csGRAY")))
```
Finally, I produced four variants of this image while employing the above mentioned transformations and on each of them applied the  `pytesseract.image_to_string(img, lang="bul")` method. 

```python
img = test_imgs[3]

imgs_transf = []
# ORIG IMG
imgs_transf.append(img)
# DILATION
kernel = np.ones((1, 1), np.uint8)
img_dil = cv2.dilate(img, kernel, iterations=1)
imgs_transf.append(img_dil)
# EROSION
kernel = np.ones((2, 2), np.uint8)
img_er = cv2.erode(img, kernel, iterations=1)
imgs_transf.append(img_er)
# CLOSING
kernel = np.ones((1, 1), np.uint8)
img_clo = closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
imgs_transf.append(img_clo)

fig, axs = plt.subplots(4, 2, figsize=(15,20), tight_layout=True)

for img, ax_pair, title in zip(imgs_transf, axs, ["original", "dilation", "erosion", "closing"]):
    ax_pair[0].imshow(img)
    ax_pair[0].axis("off")
    ax_pair[0].set_title(title)
    txt = pytesseract.image_to_string(img, lang="bul")
    ax_pair[1].text(0, 0, txt, fontsize=12) 
    ax_pair[1].axis("off")
```


![](hhttps://sciencedata.dk/shared/10bf6b65c3fb544099bb78b1cc226406?download)



### Google Cloud Compute Engine

To proceed further, I started a project with a virtual machine instance on Google Cloud Plattform enabling me faster computing. After some configuration, combining the code from above, I was able to run there the script in file `data/ocr-cyr.py`. It takes pdfs files directly from one folder on sciencedata.dk and returns their textual content  to a different folder there as well (`OCR/outputs`). The files are structured as very simple jsons:

```json
{"page 1" : "recognazed text from first page", "page 2": "recognized text from second page", ...} 
```

To call it back into python, you can use `data/read_ocr_json.ipynb`. doing basically this:

```python
import sddk
conf = sddk.configure_session_and_url("SDAM_root", "648597@au.dk")
ocr_dict = sddk.read_file("/SDAM_data/OCR/outputs/AOR'1973_1972.json", "dict", conf)
ocr_dict.keys()
>>> dict_keys(['page 1', 'page 2', 'page 3', 'page 4', 'page 5', ...])
ocr_dict["page 6"]
>>> 'П. ДЕТЕВ (ПЛОВДИВ)\n\nРАЗКОПКИ НА СЕЛИЩНАТА МОГИЛА "МАЛТЕПЕ"\nПРИ С....'         
```
As expected, ocr analysis is a very time consuming process, even with a rather powerful virtual machine. To analyze 5 files in cyrilic took almost 1 hour.
