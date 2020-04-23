from PIL import Image
import pytesseract

im = Image.open("data/test-ces.png")
text = pytesseract.image_to_string(im, lang="ces",)
print(text)