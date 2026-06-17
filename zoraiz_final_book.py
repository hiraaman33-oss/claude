import os
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
import io

W, H = A4
IMG_DIR = "/home/user/claude/zoraiz_new_images/WhatsApp Unknown 2026-06-17 at 17.54.12"

# Map sorted filenames (by time) to page descriptions
# 25 files sorted:
files_sorted = sorted(os.listdir(IMG_DIR))
print("Files:")
for i, f in enumerate(files_sorted):
    print(f"  {i+1:02d}. {f}")

# Based on the order photos were described in conversation:
# 1  picnic mama
# 2  electric car park
# 3  spiderman
# 4  icecream winter xmas tree
# 5  toys mama dog
# 6  slide friend garden
# 7  piano
# 8  studying bed
# 9  birthday nonna cake
# 10 pizza restaurant
# 11 going to school backpack
# 12 cruise ship mama
# 13 1 year old sleeping pram
# 14 newborn baby
# 15 crying wants car
# 16 matching outfit baba
# 17 train baba
# 18 mama baba happy (couple)
# 19 2nd birthday baba surprise
# 20 grandparents (mama side) birthday cake
# 21 mama + ami abu forest
# 22 ama aba phoppo mall
# 23 mama baby pakistani dress

PAGES = [
    dict(img=files_sorted[0],  line1="Zoraiz e la mama fanno un picnic.",   line2="Mangiano gli snack insieme!",              bg="#FFF8E1", acc="#F9A825"),
    dict(img=files_sorted[1],  line1="Zoraiz guida la macchinina nel parco.",line2="Che divertimento!",                        bg="#E3F2FD", acc="#1565C0"),
    dict(img=files_sorted[2],  line1="Zoraiz incontra Spider-Man!",          line2="Zoraiz ama Spider-Man!",                   bg="#FFEBEE", acc="#C62828"),
    dict(img=files_sorted[3],  line1="Zoraiz mangia il gelato.",             line2="Anche d'inverno, che freddo!",             bg="#E8F5E9", acc="#2E7D32"),
    dict(img=files_sorted[4],  line1="Zoraiz e la mama mostrano i giocattoli.", line2="Zoraiz vuole il cagnolino!",            bg="#FCE4EC", acc="#AD1457"),
    dict(img=files_sorted[5],  line1="Zoraiz gioca allo scivolo con l'amico.", line2="Che bello il giardino!",                bg="#E8F5E9", acc="#388E3C"),
    dict(img=files_sorted[6],  line1="Zoraiz suona il pianoforte.",          line2="Zoraiz ama la musica!",                   bg="#EDE7F6", acc="#512DA8"),
    dict(img=files_sorted[7],  line1="Zoraiz studia e scrive sul libro.",    line2="Bravo Zoraiz!",                           bg="#E0F7FA", acc="#00838F"),
    dict(img=files_sorted[8],  line1="È il compleanno di Zoraiz!",           line2="La nonna porta la torta. Auguri!",        bg="#FFF9C4", acc="#F57F17"),
    dict(img=files_sorted[9],  line1="Zoraiz mangia la pizza.",              line2="La pizza è il piatto preferito di Zoraiz!",bg="#FBE9E7", acc="#BF360C"),
    dict(img=files_sorted[10], line1="Zoraiz va a scuola.",                  line2="Che bello lo zaino blu!",                 bg="#E3F2FD", acc="#1565C0"),
    dict(img=files_sorted[11], line1="Zoraiz e la mama sono sulla nave.",    line2="Guardano il mare insieme!",               bg="#E0F7FA", acc="#006064"),
    dict(img=files_sorted[12], line1="Zoraiz ha un anno e dorme nel passeggino.", line2="Che tenero!",                        bg="#EDE7F6", acc="#4527A0"),
    dict(img=files_sorted[13], line1="Benvenuto Zoraiz! Sei appena nato.",   line2="Zoraiz vuole il latte della mama!",       bg="#FCE4EC", acc="#C62828"),
    dict(img=files_sorted[14], line1="Zoraiz piange perché vuole la macchina.", line2="La mama lo consola!",                  bg="#FFFDE7", acc="#F57F17"),
    dict(img=files_sorted[15], line1="Zoraiz e baba hanno lo stesso vestito.", line2="Baba vuole tanto bene a Zoraiz!",      bg="#ECEFF1", acc="#37474F"),
    dict(img=files_sorted[16], line1="Zoraiz e baba aspettano il treno.",    line2="Zoraiz abbraccia il suo baba!",          bg="#E8F5E9", acc="#1B5E20"),
    dict(img=files_sorted[17], line1="La mama e il baba di Zoraiz sono felici.", line2="Ti vogliamo tanto bene, Zoraiz!",    bg="#FCE4EC", acc="#AD1457"),
    dict(img=files_sorted[18], line1="Zoraiz compie 2 anni!",               line2="Il baba gli fa la sorpresa. Auguri!",    bg="#FFF9C4", acc="#E65100"),
    dict(img=files_sorted[19], line1="Il nonno e la nonna festeggiano.",     line2="Zoraiz li ama tanto!",                   bg="#EFEBE9", acc="#4E342E"),
    dict(img=files_sorted[20], line1="Questi sono i nonni di Zoraiz.",       line2="La nonna e il nonno lo amano tanto!",    bg="#F1F8E9", acc="#33691E"),
    dict(img=files_sorted[21], line1="Il nonno, la nonna e la zia di Zoraiz.", line2="Lo amano tanto!",                     bg="#FFF3E0", acc="#E65100"),
    dict(img=files_sorted[22], line1="Zoraiz e la mama in vestito pakistano.", line2="Che belli insieme!",                   bg="#FCE4EC", acc="#880E4F"),
    dict(img=files_sorted[23], line1="Zoraiz va sul treno con i genitori.",  line2="Zoraiz ama il suo baba!",               bg="#E8F5E9", acc="#1B5E20"),
    dict(img=files_sorted[24], line1="La famiglia di Zoraiz è felice.",      line2="Vi voglio bene, mama e baba!",           bg="#FCE4EC", acc="#AD1457"),
]


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


def load_photo(filename, size=800):
    path = os.path.join(IMG_DIR, filename)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # smart crop: for portrait photos show upper portion (faces), for landscape centre
    side = min(w, h)
    left = (w - side) // 2
    if h > w:  # portrait - take from top 60% to get faces
        top = 0
    else:
        top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def rounded_corners(img, radius=60):
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, img.size[0]-1, img.size[1]-1], radius=radius, fill=255)
    img = img.convert("RGBA")
    img.putalpha(mask)
    return img


def to_reader(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def draw_cover(c):
    # gradient background: orange top → pink bottom
    from reportlab.lib.colors import HexColor
    c.setFillColor(HexColor("#FF8F00"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    # light overlay bottom
    c.setFillColor(HexColor("#FCE4EC"))
    c.roundRect(0, 0, W, H*0.45, 0, fill=1, stroke=0)

    # top banner
    c.setFillColor(HexColor("#FF8F00"))
    c.roundRect(18, H-148, W-36, 132, 24, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 48)
    c.drawCentredString(W/2, H-88, "Il Mio Libro")
    c.setFont("Helvetica-Bold", 34)
    c.drawCentredString(W/2, H-130, "di Zoraiz")

    # use first real photo as cover
    try:
        photo = load_photo(files_sorted[15], size=860)  # matching outfit - great cover photo
        photo_r = rounded_corners(photo, radius=90)
        photo_r = photo_r.resize((380, 380), Image.LANCZOS)
        px = (W - 380) / 2
        py = H - 148 - 380 - 20
        # shadow
        c.setFillColor(HexColor("#BDBDBD"))
        c.roundRect(px+7, py-7, 380, 380, 22, fill=1, stroke=0)
        # gold border
        c.setStrokeColor(HexColor("#FF8F00"))
        c.setLineWidth(6)
        c.setFillColor(white)
        c.roundRect(px-10, py-10, 400, 400, 26, fill=1, stroke=1)
        c.drawImage(to_reader(photo_r), px, py, width=380, height=380, mask='auto')
    except Exception as e:
        print(f"Cover photo error: {e}")

    # bottom box
    c.setFillColor(HexColor("#FF8F00"))
    c.roundRect(22, 38, W-44, 88, 18, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 21)
    c.drawCentredString(W/2, 100, "Impariamo l'italiano insieme!")
    c.setFont("Helvetica", 17)
    c.drawCentredString(W/2, 68, f"{len(PAGES)} pagine di avventure")
    # hearts
    for hx in [55, W-55]:
        c.setFont("Helvetica-Bold", 22)
        c.drawCentredString(hx, 82, "♥")


def draw_page(c, page, num, total):
    bg = hex_rgb(page["bg"])
    acc = hex_rgb(page["acc"])

    c.setFillColorRGB(*bg)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # top bar
    c.setFillColorRGB(*acc)
    c.rect(0, H-56, W, 56, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(W/2, H-36, "Il Mio Libro di Zoraiz")

    # page number circle
    c.setFillColorRGB(*bg)
    c.circle(W-36, H-28, 18, fill=1, stroke=0)
    c.setFillColorRGB(*acc)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(W-36, H-33, str(num))

    # photo
    try:
        photo = load_photo(page["img"], size=860)
        photo_r = rounded_corners(photo, radius=80)
        img_size = 400
        photo_r = photo_r.resize((img_size, img_size), Image.LANCZOS)
        ix = (W - img_size) / 2
        iy = H - 60 - img_size - 8

        # shadow
        c.setFillColorRGB(0.72, 0.72, 0.72)
        c.roundRect(ix+7, iy-7, img_size, img_size, 20, fill=1, stroke=0)
        # white frame + accent border
        c.setFillColor(white)
        c.setStrokeColorRGB(*acc)
        c.setLineWidth(5)
        c.roundRect(ix-10, iy-10, img_size+20, img_size+20, 24, fill=1, stroke=1)
        c.drawImage(to_reader(photo_r), ix, iy, width=img_size, height=img_size, mask='auto')
    except Exception as e:
        print(f"  Page {num} photo error: {e}")
        iy = H - 60 - 400 - 8

    # sentence box
    bx = 22; by = iy - 122; bw = W - 44; bh = 110
    c.setFillColorRGB(*acc)
    c.roundRect(bx, by, bw, bh, 16, fill=1, stroke=0)
    c.setFillColor(white)
    line1 = page["line1"]
    line2 = page.get("line2", "")
    if line2:
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(W/2, by + bh - 36, line1)
        c.setFont("Helvetica-Bold", 22)
        c.drawCentredString(W/2, by + bh - 68, line2)
    else:
        c.setFont("Helvetica-Bold", 26)
        c.drawCentredString(W/2, by + bh - 48, line1)

    # progress dots
    for i in range(total):
        dx = W/2 - (total-1)*7 + i*14
        if i == num-1:
            c.setFillColorRGB(*acc)
            c.circle(dx, 24, 6, fill=1, stroke=0)
        else:
            c.setFillColorRGB(*bg)
            c.setStrokeColorRGB(*acc)
            c.setLineWidth(1.5)
            c.circle(dx, 24, 4, fill=1, stroke=1)


def make_pdf(path):
    c = canvas.Canvas(path, pagesize=A4)
    draw_cover(c)
    c.showPage()
    for i, page in enumerate(PAGES):
        print(f"  Page {i+1}: {page['img']}")
        draw_page(c, page, i+1, len(PAGES))
        c.showPage()
    c.save()
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    make_pdf("/home/user/claude/zoraiz_libro_finale.pdf")
