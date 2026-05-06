import qrcode

url = "https://github.com/ayvan180/infodecay"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=20,
    border=2
)
qr.add_data(url)
qr.make(fit=True)

img = qr.make_image(fill_color="#0A1628", back_color="white")
img.save("infodecay_qr.png")
print("Saved infodecay_qr.png")