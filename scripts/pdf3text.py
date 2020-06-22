### REQUIREMENTS
import io # navigating files
from PIL import Image 
import pytesseract # requires a complex installation
from wand.image import Image as wi # working with PDFs and images

### ARGUMENTS
input_pdf = input('input filename: ')
if ".pdf" not in input_pdf:
	input_pdf = input_pdf + ".pdf"
language = input("OCR language (e.g. 'eng': " )
output_txt = input("output filename: ")
if ".txt" not in output_txt:
	output_txt = output_txt + ".txt"


### READ PDF FILE AS JPEG
pdf = wi(filename = input_pdf, resolution = 300)
pdfImg = pdf.convert('jpeg')
### TRANSFORM IT TO IMAGES 
imgBlobs = []
for img in pdfImg.sequence:
	page = wi(image = img)
	imgBlobs.append(page.make_blob('jpeg'))

### EXTRACT TEXT FROM THE IMAGES
extracted_text = []
i = 1
for imgBlob in imgBlobs:
	im = Image.open(io.BytesIO(imgBlob))
	text = pytesseract.image_to_string(im, lang = language)
	extracted_text.append(text + " [end-of-page" + str(i) + "]")
	i += 1

### SAVE THE FILE
file = open(output_txt,"w")
file.write(" ".join(extracted_text))