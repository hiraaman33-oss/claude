"""
Zoraiz's Day - Italian picture book with SVG-rendered children's book illustrations
"""
import io, os
import cairosvg
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader

W, H = A4
SZ = 500  # SVG canvas size


def svg_to_pil(svg_str, size=500):
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=size, output_height=size)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def to_reader(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


# ── SVG scene builders ────────────────────────────────────────────────────────

def child_svg(cx, cy, shirt="#4FC3F7", pants="#5C6BC0", scale=1.0, smile=True, arm_l="down", arm_r="down"):
    """Draw a cute cartoon child at position (cx, cy=feet level)."""
    s = scale
    # body proportions: head big, body short, legs short
    head_r = 38 * s
    body_h = 55 * s
    body_w = 32 * s
    leg_h = 48 * s
    arm_len = 40 * s

    hy = cy - leg_h - body_h - head_r  # head center y

    arm_angles = {"down": (110, 70), "up": (60, 120), "out": (180, 0), "forward": (90, 90)}
    la, ra = arm_angles.get(arm_l, (110,0)), arm_angles.get(arm_r, (0,70))

    import math
    def pt(cx_, cy_, angle_deg, length):
        r = math.radians(angle_deg)
        return cx_ + length*math.cos(r), cy_ + length*math.sin(r)

    lax, lay = pt(cx - body_w*0.8, hy + head_r*1.2, la[0], arm_len)
    rax, ray = pt(cx + body_w*0.8, hy + head_r*1.2, ra[1], arm_len)

    skin = "#FFCC80"
    hair = "#5D4037"
    eye_fill = "#333"
    shoe = "#4E342E"
    sock = "#FFFFFF"

    svg = f"""
  <!-- child -->
  <!-- left leg -->
  <rect x="{cx - body_w*0.55:.1f}" y="{cy - leg_h:.1f}" width="{body_w*0.42:.1f}" height="{leg_h:.1f}" rx="8" fill="{pants}"/>
  <!-- right leg -->
  <rect x="{cx + body_w*0.13:.1f}" y="{cy - leg_h:.1f}" width="{body_w*0.42:.1f}" height="{leg_h:.1f}" rx="8" fill="{pants}"/>
  <!-- left shoe -->
  <ellipse cx="{cx - body_w*0.34:.1f}" cy="{cy:.1f}" rx="{14*s:.1f}" ry="{7*s:.1f}" fill="{shoe}"/>
  <!-- right shoe -->
  <ellipse cx="{cx + body_w*0.34:.1f}" cy="{cy:.1f}" rx="{14*s:.1f}" ry="{7*s:.1f}" fill="{shoe}"/>
  <!-- body -->
  <rect x="{cx - body_w:.1f}" y="{cy - leg_h - body_h:.1f}" width="{body_w*2:.1f}" height="{body_h:.1f}" rx="14" fill="{shirt}"/>
  <!-- collar -->
  <ellipse cx="{cx:.1f}" cy="{cy - leg_h - body_h:.1f}" rx="{12*s:.1f}" ry="{8*s:.1f}" fill="{skin}"/>
  <!-- left arm -->
  <line x1="{cx - body_w*0.8:.1f}" y1="{hy + head_r*1.2:.1f}" x2="{lax:.1f}" y2="{lay:.1f}" stroke="{skin}" stroke-width="{14*s:.1f}" stroke-linecap="round"/>
  <!-- right arm -->
  <line x1="{cx + body_w*0.8:.1f}" y1="{hy + head_r*1.2:.1f}" x2="{rax:.1f}" y2="{ray:.1f}" stroke="{skin}" stroke-width="{14*s:.1f}" stroke-linecap="round"/>
  <!-- head -->
  <ellipse cx="{cx:.1f}" cy="{hy:.1f}" rx="{head_r:.1f}" ry="{head_r*1.05:.1f}" fill="{skin}"/>
  <!-- hair -->
  <ellipse cx="{cx:.1f}" cy="{hy - head_r*0.5:.1f}" rx="{head_r:.1f}" ry="{head_r*0.65:.1f}" fill="{hair}"/>
  <!-- left eye -->
  <circle cx="{cx - head_r*0.3:.1f}" cy="{hy:.1f}" r="{5*s:.1f}" fill="{eye_fill}"/>
  <circle cx="{cx - head_r*0.3 + 2*s:.1f}" cy="{hy - 2*s:.1f}" r="{1.5*s:.1f}" fill="white"/>
  <!-- right eye -->
  <circle cx="{cx + head_r*0.3:.1f}" cy="{hy:.1f}" r="{5*s:.1f}" fill="{eye_fill}"/>
  <circle cx="{cx + head_r*0.3 + 2*s:.1f}" cy="{hy - 2*s:.1f}" r="{1.5*s:.1f}" fill="white"/>
  <!-- smile -->
  {'<path d="M' + f'{cx - 12*s:.1f} {hy + 14*s:.1f}' + ' Q' + f'{cx:.1f} {hy + 22*s:.1f}' + ' ' + f'{cx + 12*s:.1f} {hy + 14*s:.1f}' + '" stroke="#E57373" stroke-width="' + f'{3*s:.1f}' + '" fill="none" stroke-linecap="round"/>' if smile else ''}
  <!-- cheeks -->
  <ellipse cx="{cx - head_r*0.55:.1f}" cy="{hy + 8*s:.1f}" rx="{8*s:.1f}" ry="{5*s:.1f}" fill="#FFAB91" opacity="0.6"/>
  <ellipse cx="{cx + head_r*0.55:.1f}" cy="{hy + 8*s:.1f}" rx="{8*s:.1f}" ry="{5*s:.1f}" fill="#FFAB91" opacity="0.6"/>
"""
    return svg


def mama_svg(cx, cy, scale=1.0):
    s = scale
    skin = "#FFCC80"
    hair = "#4E342E"
    dress = "#EC407A"
    head_r = 36 * s
    body_h = 75 * s
    leg_h = 65 * s
    bw = 30 * s
    hy = cy - leg_h - body_h - head_r

    return f"""
  <!-- mama -->
  <!-- dress/skirt -->
  <polygon points="{cx:.1f},{cy - leg_h:.1f} {cx - bw*1.6:.1f},{cy:.1f} {cx + bw*1.6:.1f},{cy:.1f}" fill="{dress}"/>
  <!-- body upper -->
  <rect x="{cx - bw:.1f}" y="{cy - leg_h - body_h:.1f}" width="{bw*2:.1f}" height="{body_h:.1f}" rx="12" fill="{dress}"/>
  <!-- arms -->
  <line x1="{cx - bw:.1f}" y1="{hy + head_r:.1f}" x2="{cx - bw*2.5:.1f}" y2="{hy + head_r + 45*s:.1f}" stroke="{skin}" stroke-width="{13*s:.1f}" stroke-linecap="round"/>
  <line x1="{cx + bw:.1f}" y1="{hy + head_r:.1f}" x2="{cx + bw*2.5:.1f}" y2="{hy + head_r + 45*s:.1f}" stroke="{skin}" stroke-width="{13*s:.1f}" stroke-linecap="round"/>
  <!-- head -->
  <ellipse cx="{cx:.1f}" cy="{hy:.1f}" rx="{head_r:.1f}" ry="{head_r*1.05:.1f}" fill="{skin}"/>
  <!-- hair long -->
  <ellipse cx="{cx:.1f}" cy="{hy + 10*s:.1f}" rx="{head_r + 5*s:.1f}" ry="{head_r*1.3:.1f}" fill="{hair}"/>
  <ellipse cx="{cx:.1f}" cy="{hy:.1f}" rx="{head_r:.1f}" ry="{head_r*1.05:.1f}" fill="{skin}"/>
  <ellipse cx="{cx:.1f}" cy="{hy - head_r*0.5:.1f}" rx="{head_r:.1f}" ry="{head_r*0.65:.1f}" fill="{hair}"/>
  <!-- eyes -->
  <circle cx="{cx - head_r*0.3:.1f}" cy="{hy:.1f}" r="{4.5*s:.1f}" fill="#333"/>
  <circle cx="{cx + head_r*0.3:.1f}" cy="{hy:.1f}" r="{4.5*s:.1f}" fill="#333"/>
  <!-- smile -->
  <path d="M{cx - 11*s:.1f} {hy + 12*s:.1f} Q{cx:.1f} {hy + 20*s:.1f} {cx + 11*s:.1f} {hy + 12*s:.1f}" stroke="#E91E63" stroke-width="{2.5*s:.1f}" fill="none" stroke-linecap="round"/>
  <!-- cheeks -->
  <ellipse cx="{cx - head_r*0.55:.1f}" cy="{hy + 7*s:.1f}" rx="{7*s:.1f}" ry="{4.5*s:.1f}" fill="#F48FB1" opacity="0.6"/>
  <ellipse cx="{cx + head_r*0.55:.1f}" cy="{hy + 7*s:.1f}" rx="{7*s:.1f}" ry="{4.5*s:.1f}" fill="#F48FB1" opacity="0.6"/>
"""


def baba_svg(cx, cy, scale=1.0):
    s = scale
    skin = "#D7A870"
    hair = "#3E2723"
    shirt = "#1565C0"
    pants = "#37474F"
    head_r = 38 * s
    body_h = 80 * s
    leg_h = 75 * s
    bw = 34 * s
    hy = cy - leg_h - body_h - head_r

    return f"""
  <!-- baba -->
  <!-- left leg -->
  <rect x="{cx - bw*0.55:.1f}" y="{cy - leg_h:.1f}" width="{bw*0.5:.1f}" height="{leg_h:.1f}" rx="8" fill="{pants}"/>
  <!-- right leg -->
  <rect x="{cx + bw*0.05:.1f}" y="{cy - leg_h:.1f}" width="{bw*0.5:.1f}" height="{leg_h:.1f}" rx="8" fill="{pants}"/>
  <!-- shoes -->
  <ellipse cx="{cx - bw*0.3:.1f}" cy="{cy:.1f}" rx="{16*s:.1f}" ry="{8*s:.1f}" fill="#263238"/>
  <ellipse cx="{cx + bw*0.3:.1f}" cy="{cy:.1f}" rx="{16*s:.1f}" ry="{8*s:.1f}" fill="#263238"/>
  <!-- body -->
  <rect x="{cx - bw:.1f}" y="{cy - leg_h - body_h:.1f}" width="{bw*2:.1f}" height="{body_h:.1f}" rx="14" fill="{shirt}"/>
  <!-- arms -->
  <line x1="{cx - bw:.1f}" y1="{hy + head_r:.1f}" x2="{cx - bw*2.5:.1f}" y2="{hy + head_r + 55*s:.1f}" stroke="{skin}" stroke-width="{15*s:.1f}" stroke-linecap="round"/>
  <line x1="{cx + bw:.1f}" y1="{hy + head_r:.1f}" x2="{cx + bw*2.5:.1f}" y2="{hy + head_r + 55*s:.1f}" stroke="{skin}" stroke-width="{15*s:.1f}" stroke-linecap="round"/>
  <!-- head -->
  <ellipse cx="{cx:.1f}" cy="{hy:.1f}" rx="{head_r:.1f}" ry="{head_r*1.0:.1f}" fill="{skin}"/>
  <!-- hair -->
  <ellipse cx="{cx:.1f}" cy="{hy - head_r*0.5:.1f}" rx="{head_r:.1f}" ry="{head_r*0.6:.1f}" fill="{hair}"/>
  <!-- eyes -->
  <circle cx="{cx - head_r*0.3:.1f}" cy="{hy:.1f}" r="{5*s:.1f}" fill="#333"/>
  <circle cx="{cx + head_r*0.3:.1f}" cy="{hy:.1f}" r="{5*s:.1f}" fill="#333"/>
  <!-- smile -->
  <path d="M{cx - 12*s:.1f} {hy + 14*s:.1f} Q{cx:.1f} {hy + 22*s:.1f} {cx + 12*s:.1f} {hy + 14*s:.1f}" stroke="#8D6E63" stroke-width="{2.5*s:.1f}" fill="none" stroke-linecap="round"/>
"""


def house_svg(x, y, w=200, h=170, wall="#FFF9C4", roof="#E53935", door_c="#795548"):
    rh = 70  # roof height
    return f"""
  <!-- house -->
  <polygon points="{x:.0f},{y + rh:.0f} {x + w/2:.0f},{y:.0f} {x + w:.0f},{y + rh:.0f}" fill="{roof}"/>
  <rect x="{x:.0f}" y="{y + rh:.0f}" width="{w:.0f}" height="{h:.0f}" rx="4" fill="{wall}" stroke="#E0E0E0" stroke-width="2"/>
  <!-- door -->
  <rect x="{x + w/2 - 20:.0f}" y="{y + rh + h - 55:.0f}" width="40" height="55" rx="6" fill="{door_c}"/>
  <circle cx="{x + w/2 + 12:.0f}" cy="{y + rh + h - 27:.0f}" r="4" fill="#FFD54F"/>
  <!-- windows -->
  <rect x="{x + 18:.0f}" y="{y + rh + 22:.0f}" width="40" height="35" rx="4" fill="#B3E5FC" stroke="#90CAF9" stroke-width="2"/>
  <line x1="{x + 38:.0f}" y1="{y + rh + 22:.0f}" x2="{x + 38:.0f}" y2="{y + rh + 57:.0f}" stroke="#90CAF9" stroke-width="1.5"/>
  <line x1="{x + 18:.0f}" y1="{y + rh + 39:.0f}" x2="{x + 58:.0f}" y2="{y + rh + 39:.0f}" stroke="#90CAF9" stroke-width="1.5"/>
  <rect x="{x + w - 58:.0f}" y="{y + rh + 22:.0f}" width="40" height="35" rx="4" fill="#B3E5FC" stroke="#90CAF9" stroke-width="2"/>
  <line x1="{x + w - 38:.0f}" y1="{y + rh + 22:.0f}" x2="{x + w - 38:.0f}" y2="{y + rh + 57:.0f}" stroke="#90CAF9" stroke-width="1.5"/>
  <line x1="{x + w - 58:.0f}" y1="{y + rh + 39:.0f}" x2="{x + w - 18:.0f}" y2="{y + rh + 39:.0f}" stroke="#90CAF9" stroke-width="1.5"/>
"""


def tree_svg(cx, base_y, trunk_h=50, crown_r=45, color="#43A047"):
    return f"""
  <rect x="{cx-9:.0f}" y="{base_y - trunk_h:.0f}" width="18" height="{trunk_h:.0f}" rx="5" fill="#795548"/>
  <circle cx="{cx:.0f}" cy="{base_y - trunk_h - crown_r*0.7:.0f}" r="{crown_r:.0f}" fill="{color}"/>
  <circle cx="{cx - crown_r*0.5:.0f}" cy="{base_y - trunk_h - crown_r*0.4:.0f}" r="{crown_r*0.7:.0f}" fill="{color}"/>
  <circle cx="{cx + crown_r*0.5:.0f}" cy="{base_y - trunk_h - crown_r*0.4:.0f}" r="{crown_r*0.7:.0f}" fill="{color}"/>
"""


def cloud_svg(cx, cy, scale=1.0):
    s = scale
    return f"""
  <ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="{45*s:.0f}" ry="{25*s:.0f}" fill="white" opacity="0.9"/>
  <ellipse cx="{cx-28*s:.0f}" cy="{cy+5*s:.0f}" rx="{28*s:.0f}" ry="{20*s:.0f}" fill="white" opacity="0.9"/>
  <ellipse cx="{cx+28*s:.0f}" cy="{cy+5*s:.0f}" rx="{28*s:.0f}" ry="{20*s:.0f}" fill="white" opacity="0.9"/>
  <ellipse cx="{cx-14*s:.0f}" cy="{cy-12*s:.0f}" rx="{22*s:.0f}" ry="{18*s:.0f}" fill="white" opacity="0.9"/>
  <ellipse cx="{cx+14*s:.0f}" cy="{cy-12*s:.0f}" rx="{22*s:.0f}" ry="{18*s:.0f}" fill="white" opacity="0.9"/>
"""


def sun_svg(cx, cy, r=40):
    rays = ""
    import math
    for a in range(0, 360, 30):
        x1 = cx + (r+8)*math.cos(math.radians(a))
        y1 = cy + (r+8)*math.sin(math.radians(a))
        x2 = cx + (r+22)*math.cos(math.radians(a))
        y2 = cy + (r+22)*math.sin(math.radians(a))
        rays += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#FFD54F" stroke-width="4" stroke-linecap="round"/>'
    return f"""
  {rays}
  <circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" fill="#FFD740"/>
  <circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r*0.7:.0f}" fill="#FFE57F"/>
"""


def wrap_svg(content, bg="#E3F2FD"):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SZ}" height="{SZ}" viewBox="0 0 {SZ} {SZ}">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{bg}"/>
      <stop offset="100%" stop-color="white" stop-opacity="0.3"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="{bg}"/>
{content}
</svg>"""


# ── Scenes ────────────────────────────────────────────────────────────────────

def scene_wake_up():
    content = f"""
  <!-- gradient sky through window -->
  <defs>
    <linearGradient id="dawn" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FF8F00"/>
      <stop offset="100%" stop-color="#FFE0B2"/>
    </linearGradient>
    <linearGradient id="wall_g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFF8E1"/>
      <stop offset="100%" stop-color="#FFF3CD"/>
    </linearGradient>
  </defs>
  <!-- wall -->
  <rect width="{SZ}" height="{SZ}" fill="url(#wall_g)"/>
  <!-- floor -->
  <rect y="380" width="{SZ}" height="120" rx="0" fill="#BCAAA4"/>
  <!-- bed frame -->
  <rect x="40" y="230" width="380" height="180" rx="18" fill="#795548"/>
  <!-- mattress -->
  <rect x="50" y="185" width="360" height="210" rx="14" fill="#FAFAFA"/>
  <!-- pillow -->
  <rect x="60" y="190" width="140" height="80" rx="12" fill="#B3C8F5"/>
  <!-- blanket -->
  <rect x="50" y="290" width="360" height="100" rx="10" fill="#EF9A9A"/>
  <!-- blanket stripes -->
  <rect x="50" y="310" width="360" height="12" rx="4" fill="#E57373" opacity="0.5"/>
  <rect x="50" y="335" width="360" height="12" rx="4" fill="#E57373" opacity="0.5"/>
  <!-- headboard -->
  <rect x="40" y="180" width="380" height="60" rx="14" fill="#6D4C41"/>
  <!-- window with sunrise -->
  <rect x="290" y="50" width="160" height="130" rx="8" fill="url(#dawn)" stroke="#A1887F" stroke-width="4"/>
  <line x1="370" y1="50" x2="370" y2="180" stroke="#A1887F" stroke-width="3"/>
  <line x1="290" y1="115" x2="450" y2="115" stroke="#A1887F" stroke-width="3"/>
  <!-- sun in window -->
  {sun_svg(370, 90, r=28)}
  <!-- curtains -->
  <path d="M290,50 Q260,115 275,180" fill="#F48FB1" opacity="0.8"/>
  <path d="M450,50 Q480,115 465,180" fill="#F48FB1" opacity="0.8"/>
  <!-- Zoraiz waking up - sitting in bed -->
  <!-- head -->
  <ellipse cx="205" cy="210" rx="42" ry="44" fill="#FFCC80"/>
  <!-- hair -->
  <ellipse cx="205" cy="173" rx="42" ry="28" fill="#5D4037"/>
  <!-- eyes (wide open, excited) -->
  <circle cx="189" cy="208" r="7" fill="#333"/>
  <circle cx="221" cy="208" r="7" fill="#333"/>
  <circle cx="191" cy="205" r="2.5" fill="white"/>
  <circle cx="223" cy="205" r="2.5" fill="white"/>
  <!-- big smile -->
  <path d="M188 225 Q205 240 222 225" stroke="#E57373" stroke-width="3.5" fill="none" stroke-linecap="round"/>
  <!-- cheeks -->
  <ellipse cx="178" cy="220" rx="10" ry="7" fill="#FFAB91" opacity="0.7"/>
  <ellipse cx="232" cy="220" rx="10" ry="7" fill="#FFAB91" opacity="0.7"/>
  <!-- pyjama top -->
  <rect x="172" y="248" width="66" height="50" rx="10" fill="#7986CB"/>
  <!-- arms up (stretching) -->
  <line x1="172" y1="262" x2="135" y2="225" stroke="#FFCC80" stroke-width="16" stroke-linecap="round"/>
  <line x1="238" y1="262" x2="275" y2="225" stroke="#FFCC80" stroke-width="16" stroke-linecap="round"/>
  <!-- alarm clock on bedside table -->
  <rect x="20" y="280" width="55" height="80" rx="8" fill="#ECEFF1"/>
  <rect x="25" y="285" width="45" height="50" rx="6" fill="#B3C8F5"/>
  <!-- clock face -->
  <circle cx="47" cy="310" r="18" fill="white" stroke="#90A4AE" stroke-width="2"/>
  <line x1="47" y1="310" x2="47" y2="298" stroke="#333" stroke-width="2.5"/>
  <line x1="47" y1="310" x2="56" y2="310" stroke="#333" stroke-width="2"/>
  <!-- 7:00 label -->
  <text x="47" y="345" font-family="Arial" font-size="11" font-weight="bold" fill="#F57F17" text-anchor="middle">7:00</text>
  <!-- zzz floating away -->
  <text x="290" y="190" font-family="Arial" font-size="18" fill="#7986CB" opacity="0.7">z</text>
  <text x="313" y="173" font-family="Arial" font-size="23" fill="#7986CB" opacity="0.5">z</text>
  <text x="342" y="153" font-family="Arial" font-size="28" fill="#7986CB" opacity="0.3">Z</text>
"""
    return wrap_svg(content, "#FFF8E1")


def scene_hug_mama():
    content = f"""
  <defs>
    <linearGradient id="bg2" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FCE4EC"/>
      <stop offset="100%" stop-color="#FFF8E1"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#bg2)"/>
  <!-- floor -->
  <rect y="390" width="{SZ}" height="110" fill="#D7CCC8"/>
  <!-- wall stripe -->
  <rect y="0" width="{SZ}" height="15" fill="#F48FB1"/>
  <!-- big heart background -->
  <path d="M250,180 C250,145 295,120 295,165 C295,120 340,145 340,180 C340,230 250,270 250,270 C250,270 160,230 160,180 C160,145 205,120 205,165 C205,120 250,145 250,180 Z" fill="#FFCDD2" opacity="0.6"/>
  <!-- smaller hearts floating -->
  <text x="80" y="120" font-size="28" fill="#F48FB1" opacity="0.5">♥</text>
  <text x="360" y="100" font-size="22" fill="#F48FB1" opacity="0.4">♥</text>
  <text x="400" y="160" font-size="16" fill="#F48FB1" opacity="0.4">♥</text>
  <text x="60" y="200" font-size="18" fill="#F48FB1" opacity="0.4">♥</text>
  <!-- mama hugging -->
  {mama_svg(190, 430, scale=1.05)}
  <!-- zoraiz being hugged -->
  {child_svg(270, 430, shirt="#7986CB", pants="#5C6BC0", scale=0.85, arm_l="up", arm_r="up")}
  <!-- mama's arms wrapping around child - overlay -->
  <path d="M215,295 Q250,285 280,295" stroke="#FFCC80" stroke-width="16" fill="none" stroke-linecap="round" opacity="0.9"/>
"""
    return wrap_svg(content, "#FCE4EC")


def scene_bathroom():
    content = f"""
  <defs>
    <pattern id="tiles" x="0" y="0" width="50" height="50" patternUnits="userSpaceOnUse">
      <rect width="50" height="50" fill="#E0F7FA"/>
      <rect x="1" y="1" width="47" height="47" fill="#B2EBF2" rx="3"/>
    </pattern>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#tiles)"/>
  <!-- floor -->
  <rect y="390" width="{SZ}" height="110" fill="#80CBC4"/>
  <!-- mirror -->
  <rect x="100" y="50" width="220" height="170" rx="12" fill="#E0F7FA" stroke="#80DEEA" stroke-width="5"/>
  <rect x="108" y="58" width="204" height="154" rx="8" fill="#B2EBF2" opacity="0.6"/>
  <!-- reflection sparkles -->
  <text x="145" y="110" font-size="16" fill="white">✦</text>
  <text x="270" y="90" font-size="12" fill="white">✦</text>
  <!-- sink -->
  <rect x="95" y="240" width="230" height="90" rx="14" fill="#ECEFF1" stroke="#B0BEC5" stroke-width="3"/>
  <ellipse cx="210" cy="285" rx="70" ry="40" fill="#B3E5FC" stroke="#81D4FA" stroke-width="2"/>
  <!-- faucet -->
  <rect x="197" y="225" width="26" height="22" rx="5" fill="#90A4AE"/>
  <ellipse cx="210" cy="227" rx="18" ry="8" fill="#B0BEC5"/>
  <!-- water drops -->
  <ellipse cx="210" cy="250" rx="4" ry="6" fill="#81D4FA" opacity="0.8"/>
  <ellipse cx="204" cy="258" rx="3" ry="5" fill="#81D4FA" opacity="0.6"/>
  <ellipse cx="216" cy="262" rx="3" ry="5" fill="#81D4FA" opacity="0.6"/>
  <!-- toothbrush holder -->
  <rect x="360" y="230" width="50" height="80" rx="10" fill="#FFCCBC"/>
  <!-- toothbrush -->
  <rect x="375" y="220" width="18" height="75" rx="9" fill="#42A5F5"/>
  <rect x="375" y="220" width="18" height="25" rx="5" fill="#E3F2FD"/>
  <!-- soap -->
  <ellipse cx="380" cy="245" rx="14" ry="10" fill="#CE93D8"/>
  <!-- bubbles -->
  <circle cx="370" cy="210" r="8" fill="none" stroke="#81D4FA" stroke-width="2" opacity="0.7"/>
  <circle cx="390" cy="195" r="11" fill="none" stroke="#81D4FA" stroke-width="2" opacity="0.6"/>
  <circle cx="355" cy="185" r="6" fill="none" stroke="#81D4FA" stroke-width="2" opacity="0.5"/>
  <!-- Zoraiz at sink brushing teeth -->
  {child_svg(210, 430, shirt="#4FC3F7", pants="#1565C0", scale=0.9, arm_r="up")}
  <!-- toothbrush in hand -->
  <rect x="235" y="320" width="10" height="38" rx="5" fill="#42A5F5"/>
  <rect x="235" y="320" width="10" height="14" rx="3" fill="#E3F2FD"/>
"""
    return wrap_svg(content, "#E0F7FA")


def scene_breakfast():
    content = f"""
  <defs>
    <linearGradient id="bg4" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFF8E1"/>
      <stop offset="100%" stop-color="#FFF3E0"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#bg4)"/>
  <!-- floor -->
  <rect y="390" width="{SZ}" height="110" fill="#D7CCC8"/>
  <!-- window with sunny morning -->
  <rect x="340" y="30" width="140" height="120" rx="8" fill="#B3E5FC" stroke="#A1887F" stroke-width="4"/>
  {sun_svg(410, 60, r=25)}
  <line x1="410" y1="30" x2="410" y2="150" stroke="#A1887F" stroke-width="2"/>
  <line x1="340" y1="90" x2="480" y2="90" stroke="#A1887F" stroke-width="2"/>
  <!-- curtains -->
  <path d="M340,30 Q320,90 330,150" fill="#FFB74D" opacity="0.7"/>
  <path d="M480,30 Q500,90 490,150" fill="#FFB74D" opacity="0.7"/>
  <!-- table -->
  <rect x="40" y="280" width="430" height="120" rx="12" fill="#A1887F"/>
  <rect x="55" y="290" width="420" height="108" rx="8" fill="#BCAAA4"/>
  <!-- table legs -->
  <rect x="60" y="395" width="28" height="60" rx="6" fill="#8D6E63"/>
  <rect x="420" y="395" width="28" height="60" rx="6" fill="#8D6E63"/>
  <!-- tablecloth -->
  <rect x="40" y="268" width="430" height="26" rx="6" fill="#FFF9C4"/>
  <!-- plate 1 (Zoraiz) - eggs -->
  <ellipse cx="165" cy="268" rx="62" ry="22" fill="white" stroke="#E0E0E0" stroke-width="2"/>
  <ellipse cx="148" cy="267" rx="26" ry="18" fill="#FFF176"/>
  <ellipse cx="149" cy="267" rx="14" ry="13" fill="#FFD740"/>
  <ellipse cx="182" cy="267" rx="26" ry="18" fill="#FFF176"/>
  <ellipse cx="183" cy="267" rx="14" ry="13" fill="#FFD740"/>
  <!-- bread slice -->
  <rect x="235" y="250" width="55" height="40" rx="8" fill="#FFCC80"/>
  <rect x="238" y="253" width="49" height="34" rx="6" fill="#FFE0B2"/>
  <rect x="235" y="250" width="55" height="10" rx="5" fill="#FF8A65"/>
  <!-- chocolate milk cup -->
  <rect x="315" y="240" width="50" height="55" rx="10" fill="#6D4C41"/>
  <ellipse cx="340" cy="240" rx="25" ry="8" fill="#5D4037"/>
  <ellipse cx="340" cy="244" rx="22" ry="6" fill="#A1887F"/>
  <path d="M365,255 Q380,265 365,275" stroke="#6D4C41" stroke-width="5" fill="none"/>
  <!-- steam from milk -->
  <path d="M330,228 Q325,215 330,205" stroke="#BCAAA4" stroke-width="2.5" fill="none" opacity="0.6"/>
  <path d="M340,226 Q345,212 340,200" stroke="#BCAAA4" stroke-width="2.5" fill="none" opacity="0.6"/>
  <!-- Zoraiz at center -->
  {child_svg(250, 430, shirt="#FFB300", pants="#F57F17", scale=0.82)}
  <!-- Mama left -->
  {mama_svg(100, 430, scale=0.78)}
  <!-- Baba right -->
  {baba_svg(405, 430, scale=0.78)}
"""
    return wrap_svg(content, "#FFF8E1")


def scene_get_ready():
    content = f"""
  <rect width="{SZ}" height="{SZ}" fill="#F3E5F5"/>
  <!-- floor -->
  <rect y="390" width="{SZ}" height="110" fill="#D7CCC8"/>
  <!-- wardrobe -->
  <rect x="310" y="60" width="165" height="300" rx="10" fill="#A1887F" stroke="#8D6E63" stroke-width="3"/>
  <line x1="392" y1="60" x2="392" y2="360" stroke="#8D6E63" stroke-width="3"/>
  <!-- wardrobe handles -->
  <rect x="373" y="195" width="16" height="26" rx="8" fill="#FFD54F"/>
  <rect x="395" y="195" width="16" height="26" rx="8" fill="#FFD54F"/>
  <!-- clothes hanging inside (visible through gap) -->
  <rect x="322" y="80" width="60" height="8" rx="4" fill="#8D6E63"/>
  <path d="M340,88 L340,140 Q340,155 355,155 Q370,155 370,140 L370,88" fill="#EF9A9A" stroke="#E57373" stroke-width="1"/>
  <!-- mirror -->
  <rect x="30" y="50" width="160" height="250" rx="12" fill="#E1F5FE" stroke="#81D4FA" stroke-width="5"/>
  <!-- reflection of child in mirror -->
  <ellipse cx="110" cy="140" rx="28" ry="29" fill="#FFCC80" opacity="0.8"/>
  <ellipse cx="110" cy="118" rx="28" ry="18" fill="#5D4037" opacity="0.8"/>
  <!-- backpack on floor -->
  <rect x="340" y="325" width="90" height="80" rx="12" fill="#1E88E5"/>
  <rect x="348" y="325" width="74" height="18" rx="8" fill="#1565C0"/>
  <ellipse cx="385" cy="367" rx="16" ry="14" fill="#1565C0"/>
  <path d="M340,340 Q325,360 340,370" stroke="#1565C0" stroke-width="8" fill="none" stroke-linecap="round"/>
  <path d="M430,340 Q445,360 430,370" stroke="#1565C0" stroke-width="8" fill="none" stroke-linecap="round"/>
  <!-- star on backpack -->
  <text x="373" y="375" font-size="22" fill="#FFD740">★</text>
  <!-- Zoraiz getting dressed, looking at mirror -->
  {child_svg(250, 430, shirt="#E53935", pants="#1A237E", scale=1.0, arm_l="out", arm_r="out")}
"""
    return wrap_svg(content, "#F3E5F5")


def scene_go_to_school():
    content = f"""
  <defs>
    <linearGradient id="sky6" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#BBDEFB"/>
      <stop offset="60%" stop-color="#E3F2FD"/>
      <stop offset="100%" stop-color="#C8E6C9"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#sky6)"/>
  <!-- ground -->
  <rect y="360" width="{SZ}" height="140" fill="#81C784"/>
  <!-- pavement path -->
  <rect x="160" y="360" width="120" height="140" fill="#CFD8DC"/>
  <!-- school building -->
  {house_svg(220, 110, w=240, h=180, wall="#FFF9C4", roof="#F44336")}
  <!-- school sign -->
  <rect x="240" y="155" width="200" height="30" rx="6" fill="white" stroke="#E0E0E0" stroke-width="1"/>
  <text x="340" y="175" font-family="Arial" font-size="14" font-weight="bold" fill="#1565C0" text-anchor="middle">SCUOLA</text>
  <!-- Italian flag -->
  <rect x="430" y="90" width="45" height="30" fill="#009246"/>
  <rect x="445" y="90" width="15" height="30" fill="white"/>
  <rect x="460" y="90" width="15" height="30" fill="#CE2B37"/>
  <line x1="430" y1="70" x2="430" y2="120" stroke="#A1887F" stroke-width="3"/>
  {sun_svg(65, 65, r=40)}
  {cloud_svg(320, 55, 0.85)}
  {cloud_svg(100, 80, 0.7)}
  {tree_svg(60, 365, trunk_h=50, crown_r=48, color="#388E3C")}
  {tree_svg(460, 365, trunk_h=45, crown_r=42, color="#43A047")}
  <!-- Zoraiz walking with backpack -->
  {child_svg(210, 430, shirt="#E53935", pants="#1A237E", scale=0.95, arm_l="down", arm_r="down")}
  <!-- backpack on back -->
  <rect x="224" y="310" width="32" height="50" rx="8" fill="#1E88E5"/>
  <text x="228" y="340" font-size="18" fill="#FFD740">★</text>
"""
    return wrap_svg(content, "#E3F2FD")


def scene_maestra():
    content = f"""
  <rect width="{SZ}" height="{SZ}" fill="#FFFDE7"/>
  <!-- floor -->
  <rect y="390" width="{SZ}" height="110" fill="#D7CCC8"/>
  <!-- blackboard -->
  <rect x="40" y="50" width="420" height="200" rx="10" fill="#2E7D32" stroke="#6D4C41" stroke-width="8"/>
  <rect x="52" y="62" width="396" height="176" rx="6" fill="#388E3C"/>
  <!-- board frame decoration -->
  <rect x="40" y="238" width="420" height="16" rx="0" fill="#8D6E63"/>
  <!-- chalk writings on board -->
  <text x="80" y="110" font-family="Arial" font-size="20" font-weight="bold" fill="#FFFDE7">Ciao!</text>
  <text x="80" y="145" font-family="Arial" font-size="16" fill="#FFFDE7">A - B - C</text>
  <text x="80" y="180" font-family="Arial" font-size="16" fill="#FFFDE7">1 + 1 = 2</text>
  <!-- sun drawing on board -->
  <circle cx="360" cy="140" r="28" fill="#FFD740" opacity="0.9"/>
  <line x1="360" y1="100" x2="360" y2="88" stroke="#FFD740" stroke-width="4"/>
  <line x1="360" y1="180" x2="360" y2="192" stroke="#FFD740" stroke-width="4"/>
  <line x1="320" y1="140" x2="308" y2="140" stroke="#FFD740" stroke-width="4"/>
  <line x1="400" y1="140" x2="412" y2="140" stroke="#FFD740" stroke-width="4"/>
  <!-- chalk tray -->
  <rect x="40" y="238" width="420" height="22" rx="4" fill="#795548"/>
  <rect x="70" y="242" width="30" height="12" rx="3" fill="white"/>
  <rect x="115" y="244" width="25" height="10" rx="3" fill="#FFCCBC"/>
  <!-- Maestra Maria at board with pointer -->
  {mama_svg(120, 415, scale=1.0)}
  <!-- pointer stick -->
  <line x1="148" y1="300" x2="200" y2="145" stroke="#795548" stroke-width="5" stroke-linecap="round"/>
  <!-- desk for Zoraiz -->
  <rect x="280" y="330" width="160" height="18" rx="6" fill="#A1887F"/>
  <rect x="290" y="348" width="30" height="50" rx="5" fill="#8D6E63"/>
  <rect x="405" y="348" width="30" height="50" rx="5" fill="#8D6E63"/>
  <!-- book on desk -->
  <rect x="295" y="308" width="130" height="26" rx="4" fill="#E53935"/>
  <line x1="360" y1="308" x2="360" y2="334" stroke="#C62828" stroke-width="2"/>
  <!-- Zoraiz sitting at desk (smaller) -->
  {child_svg(355, 415, shirt="#4FC3F7", pants="#1565C0", scale=0.78)}
"""
    return wrap_svg(content, "#FFFDE7")


def scene_study():
    content = f"""
  <rect width="{SZ}" height="{SZ}" fill="#E8F5E9"/>
  <!-- floor -->
  <rect y="390" width="{SZ}" height="110" fill="#A5D6A7"/>
  <!-- window -->
  <rect x="370" y="30" width="110" height="100" rx="8" fill="#B3E5FC" stroke="#A1887F" stroke-width="4"/>
  {sun_svg(425, 55, r=22)}
  <!-- 3 desks -->
  <!-- desk 1 -->
  <rect x="30" y="295" width="130" height="15" rx="6" fill="#A1887F"/>
  <rect x="40" y="310" width="24" height="45" rx="4" fill="#8D6E63"/>
  <rect x="133" y="310" width="24" height="45" rx="4" fill="#8D6E63"/>
  <rect x="35" y="270" width="120" height="30" rx="4" fill="#EF9A9A"/>
  <!-- desk 2 -->
  <rect x="185" y="295" width="130" height="15" rx="6" fill="#A1887F"/>
  <rect x="195" y="310" width="24" height="45" rx="4" fill="#8D6E63"/>
  <rect x="288" y="310" width="24" height="45" rx="4" fill="#8D6E63"/>
  <rect x="190" y="270" width="120" height="30" rx="4" fill="#FFF59D"/>
  <!-- desk 3 -->
  <rect x="340" y="295" width="130" height="15" rx="6" fill="#A1887F"/>
  <rect x="350" y="310" width="24" height="45" rx="4" fill="#8D6E63"/>
  <rect x="443" y="310" width="24" height="45" rx="4" fill="#8D6E63"/>
  <rect x="345" y="270" width="120" height="30" rx="4" fill="#B3E5FC"/>
  <!-- 3 kids sitting and studying -->
  {child_svg(95, 420, shirt="#E53935", pants="#1A237E", scale=0.75)}
  {child_svg(250, 420, shirt="#7E57C2", pants="#4527A0", scale=0.75)}
  {child_svg(405, 420, shirt="#00ACC1", pants="#006064", scale=0.75)}
  <!-- pencils -->
  <rect x="110" y="272" width="8" height="30" rx="4" fill="#FFD740" transform="rotate(20 110 272)"/>
  <rect x="265" y="272" width="8" height="30" rx="4" fill="#EF5350" transform="rotate(-15 265 272)"/>
"""
    return wrap_svg(content, "#E8F5E9")


def scene_play_school():
    content = f"""
  <defs>
    <linearGradient id="sky9" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#B3E5FC"/>
      <stop offset="70%" stop-color="#E3F2FD"/>
      <stop offset="100%" stop-color="#C8E6C9"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#sky9)"/>
  <rect y="360" width="{SZ}" height="140" fill="#66BB6A"/>
  {sun_svg(60, 65, r=38)}
  {cloud_svg(310, 60, 0.9)}
  <!-- slide structure -->
  <rect x="340" y="155" width="25" height="195" rx="8" fill="#90A4AE"/>
  <rect x="155" y="155" width="25" height="25" rx="5" fill="#90A4AE"/>
  <!-- slide surface -->
  <path d="M175,155 L365,155 L365,180 L180,345 L155,345 Z" fill="#42A5F5"/>
  <path d="M178,155 L368,155" stroke="#1565C0" stroke-width="3"/>
  <!-- ladder rungs -->
  <rect x="330" y="180" width="40" height="10" rx="4" fill="#78909C"/>
  <rect x="330" y="210" width="40" height="10" rx="4" fill="#78909C"/>
  <rect x="330" y="240" width="40" height="10" rx="4" fill="#78909C"/>
  <rect x="330" y="270" width="40" height="10" rx="4" fill="#78909C"/>
  <rect x="330" y="300" width="40" height="10" rx="4" fill="#78909C"/>
  <!-- swing set -->
  <rect x="30" y="120" width="16" height="230" rx="6" fill="#8D6E63"/>
  <rect x="150" y="120" width="16" height="230" rx="6" fill="#8D6E63"/>
  <rect x="30" y="118" width="136" height="16" rx="6" fill="#8D6E63"/>
  <!-- swing ropes -->
  <line x1="58" y1="134" x2="58" y2="250" stroke="#A1887F" stroke-width="3"/>
  <line x1="138" y1="134" x2="138" y2="250" stroke="#A1887F" stroke-width="3"/>
  <!-- swing seat -->
  <rect x="50" y="248" width="96" height="16" rx="8" fill="#EF9A9A"/>
  <!-- Zoraiz on swing -->
  {child_svg(97, 340, shirt="#FF7043", pants="#BF360C", scale=0.75, arm_l="out", arm_r="out")}
  <!-- friend running to slide -->
  {child_svg(260, 430, shirt="#66BB6A", pants="#1B5E20", scale=0.8, arm_l="up", arm_r="down")}
  {tree_svg(460, 360, crown_r=35, color="#2E7D32")}
"""
    return wrap_svg(content, "#E8F5E9")


def scene_come_home():
    content = f"""
  <defs>
    <linearGradient id="sky10" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#B3E5FC"/>
      <stop offset="100%" stop-color="#E8F5E9"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#sky10)"/>
  <rect y="355" width="{SZ}" height="145" fill="#81C784"/>
  <!-- path -->
  <rect x="145" y="355" width="130" height="145" fill="#D7CCC8"/>
  {sun_svg(430, 65, r=35)}
  {cloud_svg(150, 70, 0.8)}
  {house_svg(20, 130, w=230, h=185, wall="#FFF9C4", roof="#E53935")}
  <!-- home sign -->
  <text x="100" y="192" font-family="Arial" font-size="13" font-weight="bold" fill="#E53935" text-anchor="middle">🏠 CASA</text>
  {tree_svg(390, 360, crown_r=48, color="#388E3C")}
  {tree_svg(470, 360, crown_r=35, color="#43A047")}
  <!-- Zoraiz walking home with backpack -->
  {child_svg(255, 430, shirt="#E53935", pants="#1A237E", scale=0.95, arm_l="down", arm_r="down")}
  <rect x="222" y="316" width="32" height="50" rx="8" fill="#1E88E5"/>
  <text x="226" y="346" font-size="18" fill="#FFD740">★</text>
  <!-- mama waving from door -->
  {mama_svg(105, 430, scale=0.75)}
"""
    return wrap_svg(content, "#E3F2FD")


def scene_lunch():
    content = f"""
  <defs>
    <linearGradient id="bg11" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFF8E1"/>
      <stop offset="100%" stop-color="#FFF3E0"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#bg11)"/>
  <rect y="390" width="{SZ}" height="110" fill="#D7CCC8"/>
  <!-- window -->
  <rect x="20" y="30" width="110" height="100" rx="8" fill="#B3E5FC" stroke="#A1887F" stroke-width="4"/>
  {sun_svg(75, 55, r=22)}
  <!-- dining table -->
  <rect x="60" y="280" width="390" height="120" rx="12" fill="#8D6E63"/>
  <rect x="70" y="290" width="380" height="108" rx="8" fill="#A1887F"/>
  <!-- tablecloth edge -->
  <rect x="60" y="268" width="390" height="26" rx="8" fill="#FFF9C4"/>
  <!-- big plate with pasta/lunch -->
  <ellipse cx="255" cy="265" rx="90" ry="28" fill="white" stroke="#E0E0E0" stroke-width="3"/>
  <!-- pasta swirls -->
  <path d="M200,262 Q215,250 230,262 Q245,274 260,262 Q275,250 290,262 Q305,274 310,265" stroke="#FF8F00" stroke-width="5" fill="none" stroke-linecap="round"/>
  <path d="M205,270 Q220,258 235,270 Q250,282 265,270 Q280,258 295,270" stroke="#FF8F00" stroke-width="5" fill="none" stroke-linecap="round"/>
  <!-- tomato sauce -->
  <circle cx="232" cy="260" r="10" fill="#EF5350" opacity="0.7"/>
  <circle cx="268" cy="265" r="8" fill="#EF5350" opacity="0.6"/>
  <!-- fork -->
  <rect x="358" y="246" width="8" height="55" rx="4" fill="#BDBDBD"/>
  <rect x="354" y="246" width="4" height="20" rx="2" fill="#BDBDBD"/>
  <rect x="362" y="246" width="4" height="20" rx="2" fill="#BDBDBD"/>
  <!-- glass of water -->
  <rect x="130" y="242" width="45" height="55" rx="8" fill="#B3E5FC" opacity="0.8" stroke="#81D4FA" stroke-width="2"/>
  <!-- Zoraiz eating -->
  {child_svg(255, 430, shirt="#FFB300", pants="#E65100", scale=0.85)}
  <!-- hand reaching to plate -->
  <line x1="285" y1="337" x2="310" y2="280" stroke="#FFCC80" stroke-width="14" stroke-linecap="round"/>
"""
    return wrap_svg(content, "#FFF8E1")


def scene_nap():
    content = f"""
  <defs>
    <linearGradient id="napbg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#C5CAE9"/>
      <stop offset="100%" stop-color="#E8EAF6"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#napbg)"/>
  <rect y="380" width="{SZ}" height="120" fill="#BCAAA4"/>
  <!-- curtained window - dim afternoon -->
  <rect x="340" y="30" width="140" height="130" rx="8" fill="#7986CB" opacity="0.8" stroke="#A1887F" stroke-width="4"/>
  <!-- curtains closed -->
  <rect x="340" y="30" width="75" height="130" rx="4" fill="#9C27B0" opacity="0.7"/>
  <rect x="405" y="30" width="75" height="130" rx="4" fill="#9C27B0" opacity="0.7"/>
  <!-- moon/stars through gap -->
  <circle cx="418" cy="95" r="16" fill="#FFF9C4" opacity="0.8"/>
  <!-- bed frame -->
  <rect x="30" y="215" width="450" height="200" rx="20" fill="#6D4C41"/>
  <!-- mattress -->
  <rect x="42" y="165" width="428" height="235" rx="16" fill="#FAFAFA"/>
  <!-- headboard -->
  <rect x="30" y="155" width="450" height="70" rx="16" fill="#5D4037"/>
  <!-- pillow -->
  <rect x="55" y="170" width="175" height="85" rx="14" fill="#B3C8F5"/>
  <!-- blanket with stars -->
  <rect x="42" y="278" width="428" height="115" rx="12" fill="#7986CB"/>
  <!-- star pattern on blanket -->
  <text x="90" y="320" font-size="18" fill="#C5CAE9">★</text>
  <text x="160" y="340" font-size="14" fill="#C5CAE9">★</text>
  <text x="240" y="315" font-size="20" fill="#C5CAE9">★</text>
  <text x="320" y="335" font-size="16" fill="#C5CAE9">★</text>
  <text x="400" y="310" font-size="18" fill="#C5CAE9">★</text>
  <!-- Zoraiz sleeping - head on pillow -->
  <ellipse cx="175" cy="210" rx="50" ry="52" fill="#FFCC80"/>
  <!-- hair -->
  <ellipse cx="175" cy="172" rx="50" ry="30" fill="#5D4037"/>
  <!-- closed eyes -->
  <path d="M152,207 Q163,215 174,207" stroke="#5D4037" stroke-width="3" fill="none"/>
  <path d="M176,207 Q187,215 198,207" stroke="#5D4037" stroke-width="3" fill="none"/>
  <!-- little smile -->
  <path d="M160,225 Q175,235 190,225" stroke="#FFAB91" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <!-- cheeks -->
  <ellipse cx="148" cy="218" rx="11" ry="7" fill="#FFAB91" opacity="0.6"/>
  <ellipse cx="202" cy="218" rx="11" ry="7" fill="#FFAB91" opacity="0.6"/>
  <!-- Zzz bubbles -->
  <text x="240" y="175" font-family="Arial" font-size="22" fill="#9FA8DA" opacity="0.8">z</text>
  <text x="270" y="150" font-family="Arial" font-size="28" fill="#9FA8DA" opacity="0.6">z</text>
  <text x="308" y="120" font-family="Arial" font-size="34" fill="#9FA8DA" opacity="0.4">Z</text>
  <!-- teddy bear -->
  <circle cx="415" cy="255" r="35" fill="#BCAAA4"/>
  <circle cx="390" cy="228" r="20" fill="#BCAAA4"/>
  <circle cx="440" cy="228" r="20" fill="#BCAAA4"/>
  <circle cx="395" cy="232" r="10" fill="#A1887F"/>
  <circle cx="435" cy="232" r="10" fill="#A1887F"/>
  <circle cx="415" cy="268" r="22" fill="#BCAAA4"/>
  <circle cx="415" cy="250" r="10" fill="#A1887F"/>
  <circle cx="410" cy="248" r="4" fill="#333"/>
  <circle cx="420" cy="248" r="4" fill="#333"/>
  <path d="M407,260 Q415,265 423,260" stroke="#333" stroke-width="2" fill="none"/>
"""
    return wrap_svg(content, "#E8EAF6")


def scene_park_afternoon():
    content = f"""
  <defs>
    <linearGradient id="sky13" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#81D4FA"/>
      <stop offset="60%" stop-color="#B3E5FC"/>
      <stop offset="100%" stop-color="#C8E6C9"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#sky13)"/>
  <rect y="340" width="{SZ}" height="160" fill="#66BB6A"/>
  <!-- path in park -->
  <ellipse cx="250" cy="430" rx="170" ry="50" fill="#D7CCC8"/>
  {sun_svg(430, 60, r=42)}
  {cloud_svg(130, 70, 0.95)}
  {cloud_svg(310, 45, 0.75)}
  {tree_svg(50, 345, trunk_h=60, crown_r=60, color="#2E7D32")}
  {tree_svg(430, 345, trunk_h=55, crown_r=55, color="#388E3C")}
  {tree_svg(240, 330, trunk_h=40, crown_r=38, color="#43A047")}
  <!-- bench -->
  <rect x="300" y="328" width="150" height="16" rx="6" fill="#A1887F"/>
  <rect x="310" y="328" width="150" height="10" rx="4" fill="#BCAAA4"/>
  <rect x="310" y="344" width="20" height="30" rx="4" fill="#8D6E63"/>
  <rect x="428" y="344" width="20" height="30" rx="4" fill="#8D6E63"/>
  <!-- Zoraiz running happily to park -->
  {child_svg(200, 435, shirt="#66BB6A", pants="#1B5E20", scale=1.0, arm_l="up", arm_r="down")}
  <!-- butterfly -->
  <path d="M380,200 Q360,180 370,200 Q360,220 380,200 Q400,180 390,200 Q400,220 380,200" fill="#FF80AB" opacity="0.8"/>
  <!-- flowers in grass -->
  <circle cx="100" cy="360" r="8" fill="#FFD740"/>
  <circle cx="140" cy="355" r="6" fill="#FF80AB"/>
  <circle cx="380" cy="358" r="7" fill="#FF80AB"/>
  <circle cx="420" cy="362" r="5" fill="#FFD740"/>
"""
    return wrap_svg(content, "#E8F5E9")


def scene_play_park():
    content = f"""
  <defs>
    <linearGradient id="sky14" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#81CFEA"/>
      <stop offset="70%" stop-color="#C8E6C9"/>
      <stop offset="100%" stop-color="#A5D6A7"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#sky14)"/>
  <rect y="345" width="{SZ}" height="155" fill="#81C784"/>
  {sun_svg(65, 65, r=36)}
  {cloud_svg(330, 55, 0.9)}
  {tree_svg(440, 350, crown_r=45, color="#2E7D32")}
  {tree_svg(25, 350, crown_r=40, color="#388E3C")}
  <!-- big colourful ball -->
  <circle cx="250" cy="295" r="38" fill="#EF5350"/>
  <path d="M215,280 Q250,258 285,280" stroke="white" stroke-width="3" fill="none"/>
  <path d="M215,310 Q250,332 285,310" stroke="white" stroke-width="3" fill="none"/>
  <line x1="250" y1="257" x2="250" y2="333" stroke="white" stroke-width="3"/>
  <!-- 3 kids playing with ball -->
  {child_svg(110, 430, shirt="#42A5F5", pants="#1565C0", scale=0.88, arm_r="up")}
  {child_svg(255, 430, shirt="#EF5350", pants="#B71C1C", scale=0.88, arm_l="up", arm_r="up")}
  {child_svg(390, 430, shirt="#66BB6A", pants="#1B5E20", scale=0.88, arm_l="up")}
  <!-- motion lines from ball -->
  <line x1="220" y1="270" x2="195" y2="245" stroke="#EF9A9A" stroke-width="2" stroke-dasharray="4"/>
  <!-- flowers -->
  <circle cx="170" cy="360" r="7" fill="#FFD740"/>
  <circle cx="345" cy="355" r="6" fill="#FF80AB"/>
"""
    return wrap_svg(content, "#E8F5E9")


def scene_dinner():
    content = f"""
  <defs>
    <linearGradient id="evening" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#283593"/>
      <stop offset="40%" stop-color="#5C6BC0"/>
      <stop offset="100%" stop-color="#9FA8DA"/>
    </linearGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="#FFF8E1"/>
  <rect y="390" width="{SZ}" height="110" fill="#D7CCC8"/>
  <!-- evening window with dark sky -->
  <rect x="320" y="25" width="160" height="140" rx="8" fill="url(#evening)" stroke="#A1887F" stroke-width="4"/>
  <!-- moon in window -->
  <circle cx="400" cy="70" r="24" fill="#FFF9C4"/>
  <circle cx="415" cy="62" r="18" fill="#5C6BC0"/>
  <!-- stars in window -->
  <text x="345" y="55" font-size="12" fill="white">✦</text>
  <text x="435" y="50" font-size="10" fill="white">✦</text>
  <text x="350" y="110" font-size="8" fill="white">✦</text>
  <text x="450" y="95" font-size="10" fill="white">✦</text>
  <line x1="400" y1="25" x2="400" y2="165" stroke="#A1887F" stroke-width="3"/>
  <line x1="320" y1="95" x2="480" y2="95" stroke="#A1887F" stroke-width="3"/>
  <!-- hanging lamp -->
  <line x1="250" y1="0" x2="250" y2="55" stroke="#8D6E63" stroke-width="4"/>
  <path d="M215,55 Q250,42 285,55 L280,105 Q250,115 220,105 Z" fill="#FFD54F"/>
  <ellipse cx="250" cy="105" rx="35" ry="12" fill="#FFE57F" opacity="0.6"/>
  <!-- warm light circle on table -->
  <ellipse cx="250" cy="200" rx="160" ry="50" fill="#FFF9C4" opacity="0.4"/>
  <!-- table -->
  <rect x="55" y="275" width="405" height="125" rx="12" fill="#8D6E63"/>
  <rect x="65" y="285" width="395" height="110" rx="8" fill="#A1887F"/>
  <rect x="55" y="263" width="405" height="24" rx="8" fill="#FFF9C4"/>
  <!-- 3 plates -->
  <ellipse cx="130" cy="260" rx="58" ry="20" fill="white" stroke="#E0E0E0" stroke-width="2"/>
  <ellipse cx="250" cy="260" rx="58" ry="20" fill="white" stroke="#E0E0E0" stroke-width="2"/>
  <ellipse cx="370" cy="260" rx="58" ry="20" fill="white" stroke="#E0E0E0" stroke-width="2"/>
  <!-- food on plates -->
  <ellipse cx="130" cy="258" rx="35" ry="14" fill="#FFCC80"/>
  <ellipse cx="250" cy="258" rx="35" ry="14" fill="#A5D6A7"/>
  <ellipse cx="370" cy="258" rx="35" ry="14" fill="#FFCC80"/>
  <!-- candle -->
  <rect x="242" y="225" width="16" height="35" rx="4" fill="#FFF9C4" stroke="#FFE082" stroke-width="1"/>
  <path d="M250,222 Q245,210 250,205 Q255,210 250,222" fill="#FF6F00"/>
  <!-- 3 people at table -->
  {mama_svg(105, 430, scale=0.78)}
  {child_svg(255, 430, shirt="#FFB300", pants="#E65100", scale=0.78)}
  {baba_svg(405, 430, scale=0.78)}
"""
    return wrap_svg(content, "#FFF8E1")


def scene_good_night():
    content = f"""
  <defs>
    <linearGradient id="night" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0D1B6E"/>
      <stop offset="70%" stop-color="#1A237E"/>
      <stop offset="100%" stop-color="#283593"/>
    </linearGradient>
    <radialGradient id="moon_glow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FFF9C4" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#FFF9C4" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{SZ}" height="{SZ}" fill="url(#night)"/>
  <!-- stars -->
  <text x="50" y="60" font-size="18" fill="white" opacity="0.9">✦</text>
  <text x="130" y="40" font-size="12" fill="white" opacity="0.7">✦</text>
  <text x="200" y="70" font-size="10" fill="white" opacity="0.8">✦</text>
  <text x="80" y="110" font-size="14" fill="white" opacity="0.6">✦</text>
  <text x="160" y="50" font-size="8" fill="white" opacity="0.9">✦</text>
  <text x="420" y="55" font-size="16" fill="white" opacity="0.7">✦</text>
  <text x="460" y="90" font-size="10" fill="white" opacity="0.8">✦</text>
  <text x="390" y="35" font-size="8" fill="white" opacity="0.9">✦</text>
  <text x="290" y="45" font-size="12" fill="white" opacity="0.6">✦</text>
  <!-- big moon -->
  <circle cx="370" cy="90" r="65" fill="url(#moon_glow)"/>
  <circle cx="370" cy="90" r="50" fill="#FFF9C4"/>
  <circle cx="390" cy="75" r="38" fill="#1A237E"/>
  <!-- moon face -->
  <circle cx="358" cy="88" r="5" fill="#FFF9C4"/>
  <circle cx="375" cy="100" r="4" fill="#FFF9C4"/>
  <!-- floor -->
  <rect y="365" width="{SZ}" height="135" fill="#4E342E"/>
  <!-- bed frame -->
  <rect x="25" y="215" width="465" height="200" rx="20" fill="#4E342E"/>
  <!-- mattress -->
  <rect x="38" y="165" width="440" height="235" rx="16" fill="#FAFAFA"/>
  <!-- headboard -->
  <rect x="25" y="155" width="465" height="70" rx="16" fill="#3E2723"/>
  <!-- pillow -->
  <rect x="50" y="170" width="180" height="88" rx="14" fill="#7986CB"/>
  <!-- blanket with stars -->
  <rect x="38" y="275" width="440" height="118" rx="12" fill="#283593"/>
  <text x="80" y="318" font-size="18" fill="#9FA8DA">★</text>
  <text x="155" y="338" font-size="14" fill="#9FA8DA">★</text>
  <text x="240" y="310" font-size="22" fill="#9FA8DA">★</text>
  <text x="330" y="335" font-size="16" fill="#9FA8DA">★</text>
  <text x="415" y="315" font-size="18" fill="#9FA8DA">★</text>
  <!-- Zoraiz sleeping peacefully -->
  <ellipse cx="175" cy="208" rx="52" ry="54" fill="#FFCC80"/>
  <!-- hair -->
  <ellipse cx="175" cy="168" rx="52" ry="30" fill="#5D4037"/>
  <!-- closed eyes (peaceful) -->
  <path d="M151,204 Q164,214 176,204" stroke="#5D4037" stroke-width="3" fill="none" stroke-linecap="round"/>
  <path d="M175,204 Q188,214 201,204" stroke="#5D4037" stroke-width="3" fill="none" stroke-linecap="round"/>
  <!-- little sleeping smile -->
  <path d="M160,225 Q175,234 190,225" stroke="#FFAB91" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <!-- cheeks -->
  <ellipse cx="146" cy="218" rx="12" ry="8" fill="#FFAB91" opacity="0.5"/>
  <ellipse cx="204" cy="218" rx="12" ry="8" fill="#FFAB91" opacity="0.5"/>
  <!-- Zzz -->
  <text x="245" y="178" font-family="Arial" font-size="22" fill="#7986CB" opacity="0.8">z</text>
  <text x="278" y="152" font-family="Arial" font-size="28" fill="#7986CB" opacity="0.6">z</text>
  <text x="318" y="122" font-family="Arial" font-size="36" fill="#7986CB" opacity="0.4">Z</text>
  <!-- teddy bear next to Zoraiz -->
  <circle cx="420" cy="258" r="32" fill="#A1887F"/>
  <circle cx="398" cy="234" r="18" fill="#A1887F"/>
  <circle cx="442" cy="234" r="18" fill="#A1887F"/>
  <circle cx="398" cy="238" r="9" fill="#8D6E63"/>
  <circle cx="442" cy="238" r="9" fill="#8D6E63"/>
  <circle cx="420" cy="268" r="20" fill="#A1887F"/>
  <circle cx="413" cy="250" r="4" fill="#333"/>
  <circle cx="427" cy="250" r="4" fill="#333"/>
  <path d="M412,262 Q420,268 428,262" stroke="#333" stroke-width="2" fill="none"/>
"""
    return wrap_svg(content, "#0D1B6E")


# ── Page data ──────────────────────────────────────────────────────────────────

PAGES = [
    dict(scene_fn=scene_wake_up,
         line1="Zoraiz si sveglia alle sette!",
         line2="Buongiorno Zoraiz!", bg="#FFF8E1", acc="#F9A825"),
    dict(scene_fn=scene_hug_mama,
         line1="Zoraiz corre dalla mama.",
         line2="Un abbraccio, mama!", bg="#FCE4EC", acc="#E91E63"),
    dict(scene_fn=scene_bathroom,
         line1="Zoraiz va in bagno.",
         line2="Si lava i denti, le mani e la faccia!", bg="#E0F7FA", acc="#00838F"),
    dict(scene_fn=scene_breakfast,
         line1="Colazione con mama e baba!",
         line2="Latte al cioccolato, uova e pane.", bg="#FFF3E0", acc="#E65100"),
    dict(scene_fn=scene_get_ready,
         line1="Zoraiz si veste per la scuola.",
         line2="Zaino in spalla, si parte!", bg="#F3E5F5", acc="#7B1FA2"),
    dict(scene_fn=scene_go_to_school,
         line1="Zoraiz va a scuola.",
         line2="Cammina felice!", bg="#E3F2FD", acc="#1565C0"),
    dict(scene_fn=scene_maestra,
         line1="Ciao maestra Maria!",
         line2="La maestra insegna bene.", bg="#FFFDE7", acc="#F57F17"),
    dict(scene_fn=scene_study,
         line1="Zoraiz studia con i suoi amici.",
         line2="Bravo Zoraiz!", bg="#E8F5E9", acc="#2E7D32"),
    dict(scene_fn=scene_play_school,
         line1="Zoraiz gioca con gli amici.",
         line2="Che divertimento a scuola!", bg="#E8F5E9", acc="#388E3C"),
    dict(scene_fn=scene_come_home,
         line1="Zoraiz torna a casa.",
         line2="Arrivederci amici!", bg="#E3F2FD", acc="#1565C0"),
    dict(scene_fn=scene_lunch,
         line1="Zoraiz mangia il pranzo.",
         line2="Che buono!", bg="#FBE9E7", acc="#BF360C"),
    dict(scene_fn=scene_nap,
         line1="Zoraiz fa il pisolino.",
         line2="Sogni d'oro, Zoraiz!", bg="#EDE7F6", acc="#512DA8"),
    dict(scene_fn=scene_park_afternoon,
         line1="Nel pomeriggio Zoraiz va al parco.",
         line2="Che bella giornata!", bg="#E8F5E9", acc="#2E7D32"),
    dict(scene_fn=scene_play_park,
         line1="Zoraiz gioca con i bambini.",
         line2="Che bello giocare insieme!", bg="#E3F2FD", acc="#1565C0"),
    dict(scene_fn=scene_dinner,
         line1="Zoraiz cena con mama e baba.",
         line2="La famiglia è insieme!", bg="#FFF8E1", acc="#558B2F"),
    dict(scene_fn=scene_good_night,
         line1="Zoraiz dorme.",
         line2="Buonanotte Zoraiz! Ti vogliamo bene!", bg="#1A237E", acc="#7986CB"),
]


# ── PDF drawing ────────────────────────────────────────────────────────────────

def draw_cover(c):
    c.setFillColor(HexColor("#FF8F00"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(HexColor("#FCE4EC"))
    c.roundRect(0, 0, W, H*0.42, 0, fill=1, stroke=0)

    c.setFillColor(HexColor("#FF8F00"))
    c.roundRect(18, H-158, W-36, 143, 24, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 38)
    c.drawCentredString(W/2, H-78, "La Giornata di Zoraiz")
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(W/2, H-118, "Impariamo l'italiano!")
    c.setFont("Helvetica", 16)
    c.drawCentredString(W/2, H-145, "Dal mattino alla notte!")

    try:
        svg_str = scene_wake_up()
        scene_img = svg_to_pil(svg_str, size=500)
        scene_img = scene_img.resize((370, 370), Image.LANCZOS)
        mask = Image.new("L", (370, 370), 0)
        from PIL import ImageDraw as ID
        md = ID.Draw(mask)
        md.rounded_rectangle([0,0,369,369], radius=60, fill=255)
        scene_img.putalpha(mask)
        px = (W-370)/2
        py = H-158-370-18
        c.setFillColor(HexColor("#BDBDBD"))
        c.roundRect(px+8, py-8, 370, 370, 22, fill=1, stroke=0)
        c.setStrokeColor(HexColor("#FF8F00"))
        c.setLineWidth(6)
        c.setFillColor(white)
        c.roundRect(px-10, py-10, 390, 390, 26, fill=1, stroke=1)
        buf = io.BytesIO(); scene_img.save(buf, "PNG"); buf.seek(0)
        c.drawImage(ImageReader(buf), px, py, width=370, height=370, mask='auto')
    except Exception as e:
        print(f"Cover error: {e}")

    c.setFillColor(HexColor("#FF8F00"))
    c.roundRect(22, 30, W-44, 100, 18, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(W/2, 102, "Dal mattino alla sera con Zoraiz!")
    c.setFont("Helvetica", 15)
    c.drawCentredString(W/2, 70, f"{len(PAGES)} pagine di avventure")
    for hx in [55, W-55]:
        c.setFont("Helvetica-Bold", 22)
        c.drawCentredString(hx, 78, "♥")


def draw_page(c, page, num, total):
    bg = hex_rgb(page["bg"])
    acc = hex_rgb(page["acc"])

    c.setFillColorRGB(*bg)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    c.setFillColorRGB(*acc)
    c.rect(0, H-56, W, 56, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(W/2, H-36, "La Giornata di Zoraiz")

    c.setFillColorRGB(*bg)
    c.circle(W-36, H-28, 18, fill=1, stroke=0)
    c.setFillColorRGB(*acc)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(W-36, H-33, str(num))

    try:
        svg_str = page["scene_fn"]()
        scene_img = svg_to_pil(svg_str, size=500)
        img_size = 410
        scene_img = scene_img.resize((img_size, img_size), Image.LANCZOS)
        from PIL import ImageDraw as ID
        mask = Image.new("L", (img_size, img_size), 0)
        md = ID.Draw(mask)
        md.rounded_rectangle([0,0,img_size-1,img_size-1], radius=65, fill=255)
        scene_img.putalpha(mask)
        ix = (W - img_size) / 2
        iy = H - 62 - img_size - 8
        c.setFillColorRGB(0.72, 0.72, 0.72)
        c.roundRect(ix+7, iy-7, img_size, img_size, 20, fill=1, stroke=0)
        c.setFillColor(white)
        c.setStrokeColorRGB(*acc)
        c.setLineWidth(5)
        c.roundRect(ix-10, iy-10, img_size+20, img_size+20, 24, fill=1, stroke=1)
        buf = io.BytesIO(); scene_img.save(buf, "PNG"); buf.seek(0)
        c.drawImage(ImageReader(buf), ix, iy, width=img_size, height=img_size, mask='auto')
    except Exception as e:
        print(f"Page {num} error: {e}")
        iy = H - 62 - 410 - 8

    bx = 22; by = iy - 126; bw = W - 44; bh = 114
    c.setFillColorRGB(*acc)
    c.roundRect(bx, by, bw, bh, 16, fill=1, stroke=0)
    c.setFillColor(white)
    line1 = page["line1"]
    line2 = page.get("line2", "")
    if line2:
        c.setFont("Helvetica-Bold", 23)
        c.drawCentredString(W/2, by + bh - 36, line1)
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(W/2, by + bh - 68, line2)
    else:
        c.setFont("Helvetica-Bold", 26)
        c.drawCentredString(W/2, by + bh - 48, line1)

    for i in range(total):
        dx = W/2 - (total-1)*7 + i*14
        if i == num-1:
            c.setFillColorRGB(*acc)
            c.circle(dx, 22, 6, fill=1, stroke=0)
        else:
            c.setFillColorRGB(*bg)
            c.setStrokeColorRGB(*acc)
            c.setLineWidth(1.5)
            c.circle(dx, 22, 4, fill=1, stroke=1)


def make_pdf(path):
    c = canvas.Canvas(path, pagesize=A4)
    print("Drawing cover...")
    draw_cover(c)
    c.showPage()
    for i, page in enumerate(PAGES):
        print(f"  Page {i+1}: {page['line1']}")
        draw_page(c, page, i+1, len(PAGES))
        c.showPage()
    c.save()
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    make_pdf("/home/user/claude/zoraiz_giornata.pdf")
