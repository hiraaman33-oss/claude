import os
from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.utils import ImageReader
import io

W, H = A4  # 595 x 842 pts
IMG_DIR = "/home/user/claude/zoraiz_images"

# Each page: image filename, Italian sentence (line1), Italian sentence (line2 optional)
PAGES = [
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.33 (1).jpeg",
        "line1": "Zoraiz è nel parco.",
        "line2": "Che bello il parco!",
        "emoji": "🌳",
        "bg": "#E8F5E9", "accent": "#388E3C",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.31.jpeg",
        "line1": "Zoraiz e Mamma disegnano.",
        "line2": "Che bravi!",
        "emoji": "🎨",
        "bg": "#FFF8E1", "accent": "#F57F17",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.33.jpeg",
        "line1": "Zoraiz e Babbo dormono.",
        "line2": "Buona notte!",
        "emoji": "🌙",
        "bg": "#EDE7F6", "accent": "#512DA8",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.33 (2).jpeg",
        "line1": "Zoraiz e Babbo giocano.",
        "line2": "Che divertimento!",
        "emoji": "🎣",
        "bg": "#E3F2FD", "accent": "#1565C0",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.34 (2).jpeg",
        "line1": "Zoraiz va in macchina",
        "line2": "con Mamma e Babbo.",
        "emoji": "🚗",
        "bg": "#FBE9E7", "accent": "#BF360C",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.34 (1).jpeg",
        "line1": "Zoraiz va a scuola.",
        "line2": "Bravo Zoraiz!",
        "emoji": "🏫",
        "bg": "#FFFDE7", "accent": "#F9A825",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.34 (3).jpeg",
        "line1": "Zoraiz gioca in spiaggia.",
        "line2": "Il mare è bellissimo!",
        "emoji": "🏖️",
        "bg": "#E0F7FA", "accent": "#00838F",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.34 (4).jpeg",
        "line1": "Zoraiz cammina in piazza.",
        "line2": "Che bel bambino!",
        "emoji": "👶",
        "bg": "#F3E5F5", "accent": "#6A1B9A",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.35.jpeg",
        "line1": "Zoraiz beve il succo",
        "line2": "d'arancia al mare.",
        "emoji": "🍊",
        "bg": "#FFF3E0", "accent": "#E65100",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.35 (1).jpeg",
        "line1": "Zoraiz guida il monopattino.",
        "line2": "Con il casco!",
        "emoji": "🛴",
        "bg": "#ECEFF1", "accent": "#37474F",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.36.jpeg",
        "line1": "Zoraiz salta sul trampolino.",
        "line2": "In alto Zoraiz!",
        "emoji": "🤸",
        "bg": "#FCE4EC", "accent": "#AD1457",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.36 (1).jpeg",
        "line1": "Zoraiz mangia il gelato",
        "line2": "al cioccolato. Che buono!",
        "emoji": "🍫",
        "bg": "#EFEBE9", "accent": "#4E342E",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.36 (2).jpeg",
        "line1": "Zoraiz incontra il leone.",
        "line2": "Ciao leone!",
        "emoji": "🦁",
        "bg": "#FFF8E1", "accent": "#FF8F00",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.53.jpeg",
        "line1": "Zoraiz mangia il riso.",
        "line2": "Che appetito!",
        "emoji": "🍚",
        "bg": "#F1F8E9", "accent": "#2E7D32",
    },
    {
        "img": "WhatsApp Image 2026-06-17 at 17.15.34.jpeg",
        "line1": "Zoraiz da piccolo.",
        "line2": "Com'era tenero!",
        "emoji": "👶",
        "bg": "#FCE4EC", "accent": "#C62828",
    },
]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def load_and_crop_square(path, size=800):
    """Load image, crop to square from centre, resize."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def rounded_mask(size, radius=60):
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def img_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def draw_page(c, page, page_num, total):
    bg_rgb = hex_to_rgb(page["bg"])
    acc_rgb = hex_to_rgb(page["accent"])

    # Background
    c.setFillColorRGB(*[v/255 for v in bg_rgb])
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Top stripe
    c.setFillColorRGB(*[v/255 for v in acc_rgb])
    c.rect(0, H - 58, W, 58, fill=1, stroke=0)

    # Emoji + page title in top stripe
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(W / 2, H - 38, f"Il Libro di Zoraiz")

    # Page number circle bottom-right of stripe
    c.setFillColorRGB(*[v/255 for v in bg_rgb])
    c.circle(W - 36, H - 29, 18, fill=1, stroke=0)
    c.setFillColorRGB(*[v/255 for v in acc_rgb])
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(W - 36, H - 34, str(page_num))

    # Load + process photo
    img_path = os.path.join(IMG_DIR, page["img"])
    photo = load_and_crop_square(img_path, size=900)

    # Add rounded corners
    mask = rounded_mask(900, radius=80)
    photo_rgba = photo.convert("RGBA")
    photo_rgba.putalpha(mask)

    # Drop shadow effect: draw a slightly larger dark rounded rect first
    shadow_size = 420
    photo_display = photo_rgba.resize((shadow_size, shadow_size), Image.LANCZOS)

    photo_x = (W - shadow_size) / 2
    photo_y = H - 68 - shadow_size - 10  # below top stripe

    # Shadow
    c.setFillColorRGB(0, 0, 0)
    c.setFillAlpha(0.15)
    c.roundRect(photo_x + 6, photo_y - 6, shadow_size, shadow_size, 20, fill=1, stroke=0)
    c.setFillAlpha(1.0)

    # Accent border frame
    frame_pad = 8
    c.setStrokeColorRGB(*[v/255 for v in acc_rgb])
    c.setLineWidth(5)
    c.roundRect(photo_x - frame_pad, photo_y - frame_pad,
                shadow_size + frame_pad*2, shadow_size + frame_pad*2, 24, fill=0, stroke=1)

    # White inner frame
    c.setStrokeColor(white)
    c.setLineWidth(3)
    c.roundRect(photo_x - 2, photo_y - 2,
                shadow_size + 4, shadow_size + 4, 18, fill=0, stroke=1)

    # Photo itself
    c.drawImage(ImageReader(img_to_bytes(photo_display)),
                photo_x, photo_y,
                width=shadow_size, height=shadow_size, mask='auto')

    # Sentence box
    box_y = photo_y - 130
    box_h = 115
    c.setFillColorRGB(*[v/255 for v in acc_rgb])
    c.roundRect(28, box_y, W - 56, box_h, 16, fill=1, stroke=0)

    # Italian sentences — big and bold
    c.setFillColor(white)
    line1 = page["line1"]
    line2 = page.get("line2", "")

    if line2:
        c.setFont("Helvetica-Bold", 26)
        c.drawCentredString(W / 2, box_y + box_h - 38, line1)
        c.setFont("Helvetica-Bold", 26)
        c.drawCentredString(W / 2, box_y + box_h - 72, line2)
    else:
        c.setFont("Helvetica-Bold", 28)
        c.drawCentredString(W / 2, box_y + box_h - 50, line1)

    # Bottom dots progress bar
    dot_y = 28
    for i in range(total):
        if i == page_num - 1:
            c.setFillColorRGB(*[v/255 for v in acc_rgb])
            c.circle(W/2 - (total-1)*9 + i*18, dot_y, 7, fill=1, stroke=0)
        else:
            c.setFillColorRGB(*[v/255 for v in bg_rgb])
            c.setStrokeColorRGB(*[v/255 for v in acc_rgb])
            c.setLineWidth(1.5)
            c.circle(W/2 - (total-1)*9 + i*18, dot_y, 5, fill=1, stroke=1)


def draw_cover(c):
    # Pick the best portrait shot as cover — park bench photo
    cover_img_path = os.path.join(IMG_DIR, "WhatsApp Image 2026-06-17 at 17.15.33 (1).jpeg")
    photo = Image.open(cover_img_path).convert("RGB")
    # Use full image, don't crop to square for cover
    pw, ph = photo.size
    target_w = int(W * 0.72)
    ratio = target_w / pw
    target_h = int(ph * ratio)
    if target_h > 440:
        target_h = 440
        target_w = int(pw * (target_h / ph))
    photo = photo.resize((target_w, target_h), Image.LANCZOS)

    # Rounded mask for cover photo
    mask = rounded_mask(max(target_w, target_h), radius=70)
    mask = mask.resize((target_w, target_h), Image.LANCZOS)
    photo_rgba = photo.convert("RGBA")
    photo_rgba.putalpha(mask)

    # Background gradient simulation — top yellow, bottom pink
    bg = Image.new("RGB", (int(W), int(H)), (255, 249, 196))
    draw_bg = ImageDraw.Draw(bg)
    for y in range(int(H)):
        r = int(255 - (255 - 252) * y / H)
        g = int(249 - (249 - 228) * y / H)
        b = int(196 - (196 - 230) * y / H)
        draw_bg.line([(0, y), (int(W), y)], fill=(r, g, b))
    bg_bytes = img_to_bytes(bg)
    c.drawImage(ImageReader(bg_bytes), 0, 0, width=W, height=H)

    # Decorative top banner
    c.setFillColorRGB(1.0, 0.56, 0.0)  # orange
    c.roundRect(0, H - 140, W, 140, 0, fill=1, stroke=0)

    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 46)
    c.drawCentredString(W / 2, H - 68, "Il Mio Libro")
    c.setFont("Helvetica-Bold", 34)
    c.drawCentredString(W / 2, H - 112, "di Zoraiz")

    # Stars decoration
    c.setFillColorRGB(1, 0.9, 0)
    for sx in [50, 100, W-50, W-100]:
        c.circle(sx, H - 95, 8, fill=1, stroke=0)

    # Photo centred
    px = (W - target_w) / 2
    py = H - 140 - target_h - 18

    # Shadow
    c.setFillColorRGB(0.6, 0.6, 0.6)
    c.roundRect(px + 8, py - 8, target_w, target_h, 22, fill=1, stroke=0)

    # Gold border
    c.setStrokeColorRGB(1.0, 0.56, 0.0)
    c.setLineWidth(6)
    c.roundRect(px - 10, py - 10, target_w + 20, target_h + 20, 26, fill=0, stroke=1)
    # White inner
    c.setStrokeColor(white)
    c.setLineWidth(3)
    c.roundRect(px - 3, py - 3, target_w + 6, target_h + 6, 20, fill=0, stroke=1)

    c.drawImage(ImageReader(img_to_bytes(photo_rgba)),
                px, py, width=target_w, height=target_h, mask='auto')

    # Subtitle box at bottom
    c.setFillColorRGB(1.0, 0.56, 0.0)
    c.roundRect(30, 40, W - 60, 90, 18, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(W / 2, 100, "Impariamo l'italiano insieme!")
    c.setFont("Helvetica", 17)
    c.drawCentredString(W / 2, 68, f"{len(PAGES)} pagine di avventure")

    # Little hearts
    c.setFillColorRGB(0.9, 0.1, 0.2)
    for (hx, hy) in [(65, 85), (W-65, 85)]:
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(white)
        c.drawCentredString(hx, hy - 6, "♥")


def make_pdf(output_path):
    c = canvas.Canvas(output_path, pagesize=A4)

    # Cover
    draw_cover(c)
    c.showPage()

    # Content pages
    for i, page in enumerate(PAGES):
        draw_page(c, page, i + 1, len(PAGES))
        c.showPage()

    c.save()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    make_pdf("/home/user/claude/zoraiz_libro_reale.pdf")
