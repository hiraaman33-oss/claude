"""
Zoraiz's Day - Italian picture book with illustrated scenes
Each page has a drawn illustration + Italian sentence
"""
import io
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader

W, H = A4  # 595 x 842 pts
SCENE_SIZE = 420  # pixel size of each illustration


def to_reader(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_sky(d, w, h, top_color, bot_color=None):
    if bot_color is None:
        bot_color = top_color
    for y in range(h):
        t = y / h
        r = int(top_color[0] + (bot_color[0]-top_color[0])*t)
        g = int(top_color[1] + (bot_color[1]-top_color[1])*t)
        b = int(top_color[2] + (bot_color[2]-top_color[2])*t)
        d.line([(0,y),(w,y)], fill=(r,g,b))


def draw_ground(d, w, h, y_start, color=(120,200,80)):
    d.rectangle([0, y_start, w, h], fill=color)


def draw_sun(d, cx, cy, r=40, color=(255,220,50)):
    d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
    for angle in range(0, 360, 45):
        import math
        x1 = cx + int((r+6)*math.cos(math.radians(angle)))
        y1 = cy + int((r+6)*math.sin(math.radians(angle)))
        x2 = cx + int((r+16)*math.cos(math.radians(angle)))
        y2 = cy + int((r+16)*math.sin(math.radians(angle)))
        d.line([(x1,y1),(x2,y2)], fill=(255,200,0), width=3)


def draw_moon(d, cx, cy, r=40):
    d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255,240,150))
    d.ellipse([cx-r+14, cy-r-8, cx+r+14, cy+r-8], fill=(100,120,200))


def draw_stars(d, count=8):
    import random, math
    random.seed(42)
    for _ in range(count):
        sx = random.randint(20, 380)
        sy = random.randint(20, 140)
        d.ellipse([sx-3, sy-3, sx+3, sy+3], fill=(255,255,255))


def draw_cloud(d, cx, cy, color=(240,240,255)):
    for dx, dy, r in [(-28,8,22),(0,0,28),(28,8,22),(14,-10,18),(-14,-10,18)]:
        d.ellipse([cx+dx-r, cy+dy-r, cx+dx+r, cy+dy+r], fill=color)


def draw_house(d, x, y, w=180, h=140, wall=(255,240,200), roof=(220,80,60), door=(160,100,60)):
    roof_pts = [(x, y+50), (x+w//2, y), (x+w, y+50)]
    d.polygon(roof_pts, fill=roof)
    d.rectangle([x, y+50, x+w, y+h], fill=wall)
    # door
    dw, dh = 30, 50
    dx = x + w//2 - dw//2
    d.rectangle([dx, y+h-dh, dx+dw, y+h], fill=door)
    d.ellipse([dx+dw-12, y+h-dh//2-4, dx+dw-4, y+h-dh//2+4], fill=(200,150,50))
    # windows
    for wx in [x+20, x+w-55]:
        d.rectangle([wx, y+70, wx+30, y+100], fill=(180,230,255), outline=(100,100,100), width=2)
        d.line([(wx+15, y+70),(wx+15, y+100)], fill=(100,100,100), width=1)
        d.line([(wx, y+85),(wx+30, y+85)], fill=(100,100,100), width=1)


def draw_tree(d, cx, base_y, trunk_h=40, crown_r=35, color=(60,160,60)):
    # trunk
    d.rectangle([cx-8, base_y-trunk_h, cx+8, base_y], fill=(140,90,40))
    # crown
    d.ellipse([cx-crown_r, base_y-trunk_h-crown_r*2+10, cx+crown_r, base_y-trunk_h+10], fill=color)


def draw_stick_figure(d, cx, base_y, color=(60,60,180), skin=(255,200,150), scale=1.0, facing="right"):
    s = scale
    # body
    head_r = int(18*s)
    d.ellipse([cx-head_r, base_y-int(100*s)-head_r*2, cx+head_r, base_y-int(100*s)], fill=skin, outline=(180,130,100), width=2)
    # torso
    d.rectangle([cx-int(14*s), base_y-int(100*s), cx+int(14*s), base_y-int(48*s)], fill=color)
    # legs
    d.line([(cx-int(8*s), base_y-int(48*s)),(cx-int(16*s), base_y)], fill=(80,80,80), width=int(6*s))
    d.line([(cx+int(8*s), base_y-int(48*s)),(cx+int(16*s), base_y)], fill=(80,80,80), width=int(6*s))
    # arms
    if facing == "right":
        d.line([(cx-int(14*s), base_y-int(85*s)),(cx-int(35*s), base_y-int(60*s))], fill=skin, width=int(5*s))
        d.line([(cx+int(14*s), base_y-int(85*s)),(cx+int(35*s), base_y-int(60*s))], fill=skin, width=int(5*s))
    else:
        d.line([(cx-int(14*s), base_y-int(85*s)),(cx-int(35*s), base_y-int(60*s))], fill=skin, width=int(5*s))
        d.line([(cx+int(14*s), base_y-int(85*s)),(cx+int(35*s), base_y-int(60*s))], fill=skin, width=int(5*s))
    # shoes
    d.ellipse([cx-int(22*s), base_y-int(10*s), cx-int(6*s), base_y], fill=(80,50,20))
    d.ellipse([cx+int(6*s), base_y-int(10*s), cx+int(22*s), base_y], fill=(80,50,20))


def draw_mama(d, cx, base_y, scale=1.0):
    s = scale
    head_r = int(18*s)
    skin = (255,210,160)
    dress = (220,80,120)
    # hair
    d.ellipse([cx-head_r-4, base_y-int(100*s)-head_r*2-8, cx+head_r+4, base_y-int(100*s)+10], fill=(80,40,10))
    # head
    d.ellipse([cx-head_r, base_y-int(100*s)-head_r*2, cx+head_r, base_y-int(100*s)], fill=skin)
    # dress / body
    pts = [(cx-int(14*s), base_y-int(100*s)), (cx+int(14*s), base_y-int(100*s)),
           (cx+int(28*s), base_y), (cx-int(28*s), base_y)]
    d.polygon(pts, fill=dress)
    # arms
    d.line([(cx-int(14*s), base_y-int(85*s)),(cx-int(38*s), base_y-int(55*s))], fill=skin, width=int(5*s))
    d.line([(cx+int(14*s), base_y-int(85*s)),(cx+int(38*s), base_y-int(55*s))], fill=skin, width=int(5*s))


def draw_baba(d, cx, base_y, scale=1.0):
    s = scale
    head_r = int(20*s)
    skin = (220,180,140)
    shirt = (60,100,180)
    trousers = (50,50,80)
    # head
    d.ellipse([cx-head_r, base_y-int(115*s)-head_r*2, cx+head_r, base_y-int(115*s)], fill=skin)
    # torso
    d.rectangle([cx-int(18*s), base_y-int(115*s), cx+int(18*s), base_y-int(55*s)], fill=shirt)
    # legs
    d.rectangle([cx-int(18*s), base_y-int(55*s), cx-int(4*s), base_y], fill=trousers)
    d.rectangle([cx+int(4*s), base_y-int(55*s), cx+int(18*s), base_y], fill=trousers)
    # arms
    d.line([(cx-int(18*s), base_y-int(100*s)),(cx-int(42*s), base_y-int(65*s))], fill=skin, width=int(6*s))
    d.line([(cx+int(18*s), base_y-int(100*s)),(cx+int(42*s), base_y-int(65*s))], fill=skin, width=int(6*s))
    # shoes
    d.ellipse([cx-int(26*s), base_y-int(12*s), cx-int(4*s), base_y], fill=(40,30,20))
    d.ellipse([cx+int(4*s), base_y-int(12*s), cx+int(26*s), base_y], fill=(40,30,20))


def new_scene(bg_color=(200,230,255)):
    img = Image.new("RGB", (SCENE_SIZE, SCENE_SIZE), bg_color)
    d = ImageDraw.Draw(img)
    return img, d


# ── Individual scene drawings ─────────────────────────────────────────────────

def scene_wake_up():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (255,230,150), (255,200,100))
    # bedroom wall
    d.rectangle([0, 60, SCENE_SIZE, SCENE_SIZE], fill=(240,235,220))
    # floor
    d.rectangle([0, 320, SCENE_SIZE, SCENE_SIZE], fill=(200,160,110))
    # bed
    d.rectangle([40, 200, 300, 320], fill=(180,100,60))  # frame
    d.rectangle([50, 150, 290, 310], fill=(255,255,255))  # mattress
    d.rectangle([50, 150, 290, 210], fill=(100,150,220))  # pillow
    d.rectangle([50, 240, 290, 310], fill=(220,80,80))   # blanket
    # zoraiz in bed (head peeking)
    d.ellipse([150, 165, 200, 215], fill=(255,200,150))
    # alarm clock
    d.ellipse([330, 230, 380, 280], fill=(255,80,80))
    d.ellipse([336, 236, 374, 274], fill=(255,255,255))
    d.line([(355, 255),(355, 245)], fill=(0,0,0), width=3)
    d.line([(355, 255),(365, 255)], fill=(0,0,0), width=3)
    # sun coming through window
    d.rectangle([310, 80, 400, 170], fill=(180,230,255), outline=(180,160,120), width=3)
    d.line([(355, 80),(355, 170)], fill=(180,160,120), width=2)
    d.line([(310, 125),(400, 125)], fill=(180,160,120), width=2)
    # sun rays through window
    d.polygon([(310,80),(310,120),(340,80)], fill=(255,240,150))
    # "7:00" on wall
    d.rectangle([40, 80, 120, 130], fill=(255,200,50))
    return img


def scene_hug_mama():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (255,230,200), (255,210,180))
    d.rectangle([0, 280, SCENE_SIZE, SCENE_SIZE], fill=(210,180,140))  # floor
    d.rectangle([0, 80, SCENE_SIZE, 280], fill=(255,245,235))  # wall
    # big heart
    cx, cy = 210, 200
    heart_pts = []
    import math
    for t in range(0, 360, 5):
        rad = math.radians(t)
        x = 16*(math.sin(rad)**3)
        y = -(13*math.cos(rad) - 5*math.cos(2*rad) - 2*math.cos(3*rad) - math.cos(4*rad))
        heart_pts.append((cx + int(x*6), cy + int(y*6)))
    d.polygon(heart_pts, fill=(255,100,120))
    # mama
    draw_mama(d, 160, 310, scale=1.1)
    # zoraiz (smaller, arms up hugging)
    s = 0.75
    sx, sy = 240, 310
    head_r = int(18*s)
    d.ellipse([sx-head_r, sy-int(100*s)-head_r*2, sx+head_r, sy-int(100*s)], fill=(255,200,150))
    d.rectangle([sx-int(14*s), sy-int(100*s), sx+int(14*s), sy-int(48*s)], fill=(80,160,220))
    d.line([(sx-int(8*s), sy-int(48*s)),(sx-int(16*s), sy)], fill=(80,80,80), width=4)
    d.line([(sx+int(8*s), sy-int(48*s)),(sx+int(16*s), sy)], fill=(80,80,80), width=4)
    # arms up
    d.line([(sx-int(14*s), sy-int(85*s)),(sx-int(38*s), sy-int(105*s))], fill=(255,200,150), width=4)
    d.line([(sx+int(14*s), sy-int(85*s)),(sx+int(38*s), sy-int(105*s))], fill=(255,200,150), width=4)
    return img


def scene_bathroom():
    img, d = new_scene()
    d.rectangle([0, 0, SCENE_SIZE, SCENE_SIZE], fill=(200,240,255))  # tiles bg
    # tile pattern
    for tx in range(0, SCENE_SIZE, 40):
        for ty in range(0, SCENE_SIZE, 40):
            d.rectangle([tx, ty, tx+38, ty+38], outline=(180,220,240), width=1)
    # sink
    d.rectangle([100, 200, 320, 280], fill=(240,240,240), outline=(180,180,200), width=3)
    d.ellipse([170, 230, 250, 275], fill=(200,230,250), outline=(150,150,180), width=2)
    # faucet
    d.rectangle([195, 190, 215, 210], fill=(180,180,200), width=0)
    d.ellipse([188, 180, 222, 200], fill=(200,200,220))
    # mirror above sink
    d.rectangle([110, 80, 310, 185], fill=(200,240,255), outline=(180,180,200), width=4)
    # toothbrush
    d.rectangle([340, 240, 360, 320], fill=(80,200,80))
    d.rectangle([340, 240, 360, 265], fill=(255,255,255))
    for bx in range(342, 360, 4):
        for by in range(241, 264, 4):
            d.ellipse([bx, by, bx+2, by+3], fill=(200,200,255))
    # Zoraiz at sink
    s = 0.85
    cx, sy = 210, 320
    head_r = int(18*s)
    d.ellipse([cx-head_r, sy-int(100*s)-head_r*2, cx+head_r, sy-int(100*s)], fill=(255,200,150))
    d.rectangle([cx-int(14*s), sy-int(100*s), cx+int(14*s), sy-int(48*s)], fill=(100,180,230))
    d.line([(cx-int(8*s), sy-int(48*s)),(cx-int(16*s), sy)], fill=(80,80,80), width=5)
    d.line([(cx+int(8*s), sy-int(48*s)),(cx+int(16*s), sy)], fill=(80,80,80), width=5)
    # arm reaching to sink
    d.line([(cx+int(14*s), sy-int(85*s)),(cx+int(50*s), sy-int(60*s))], fill=(255,200,150), width=5)
    # bubbles / water drops
    for bx, by in [(230,195),(250,185),(240,175),(260,200)]:
        d.ellipse([bx-5, by-5, bx+5, by+5], fill=(180,220,255,200))
    return img


def scene_breakfast():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (255,240,200), (255,220,180))
    d.rectangle([0, 200, SCENE_SIZE, SCENE_SIZE], fill=(220,190,150))  # floor
    d.rectangle([0, 80, SCENE_SIZE, 200], fill=(255,250,240))  # wall
    # table
    d.rectangle([50, 220, 380, 340], fill=(180,120,60))
    d.rectangle([60, 340, 100, 400], fill=(140,90,40))
    d.rectangle([330, 340, 370, 400], fill=(140,90,40))
    # plates
    d.ellipse([80, 200, 160, 245], fill=(255,255,255), outline=(200,200,200), width=2)
    d.ellipse([175, 200, 255, 245], fill=(255,255,255), outline=(200,200,200), width=2)
    d.ellipse([270, 200, 350, 245], fill=(255,255,255), outline=(200,200,200), width=2)
    # eggs on plate 1
    d.ellipse([95, 205, 145, 240], fill=(255,240,180))
    d.ellipse([108, 210, 132, 232], fill=(255,210,0))
    # bread
    d.rectangle([185, 205, 245, 240], fill=(240,200,120))
    d.rectangle([189, 205, 241, 218], fill=(200,150,70))
    # chocolate milk cup
    d.rectangle([280, 195, 340, 250], fill=(120,70,40))
    d.ellipse([280, 192, 340, 208], fill=(100,55,30))
    d.ellipse([284, 198, 336, 210], fill=(200,140,100))
    # handle
    d.arc([335, 205, 360, 235], 270, 90, fill=(120,70,40), width=4)
    # window
    d.rectangle([290, 85, 390, 175], fill=(180,230,255), outline=(180,160,120), width=3)
    draw_sun(d, 340, 60, r=30, color=(255,220,50))
    # Zoraiz small at center
    draw_stick_figure(d, 210, 420, color=(80,160,220), scale=0.75)
    # mama left
    draw_mama(d, 90, 420, scale=0.75)
    # baba right
    draw_baba(d, 340, 420, scale=0.75)
    return img


def scene_get_ready():
    img, d = new_scene()
    d.rectangle([0, 0, SCENE_SIZE, SCENE_SIZE], fill=(245,235,255))
    d.rectangle([0, 310, SCENE_SIZE, SCENE_SIZE], fill=(200,180,150))
    # wardrobe
    d.rectangle([280, 80, 400, 310], fill=(180,140,90), outline=(140,100,60), width=3)
    d.line([(340, 80),(340, 310)], fill=(140,100,60), width=2)
    d.ellipse([330, 185, 350, 205], fill=(200,160,80))
    d.ellipse([330, 195, 350, 215], fill=(200,160,80))
    # backpack on floor
    d.rectangle([50, 260, 130, 340], fill=(60,100,200))
    d.rectangle([60, 260, 120, 275], fill=(40,80,180))
    d.ellipse([80, 295, 100, 315], fill=(40,80,180))
    d.arc([50, 230, 130, 270], 180, 0, fill=(40,80,180), width=8)
    # Zoraiz getting dressed
    s = 1.0
    cx, sy = 180, 330
    head_r = int(18*s)
    d.ellipse([cx-head_r, sy-int(100*s)-head_r*2, cx+head_r, sy-int(100*s)], fill=(255,200,150))
    # shirt being put on
    d.rectangle([cx-int(14*s), sy-int(100*s), cx+int(14*s), sy-int(48*s)], fill=(220,60,60))
    d.line([(cx-int(8*s), sy-int(48*s)),(cx-int(16*s), sy)], fill=(50,50,80), width=6)
    d.line([(cx+int(8*s), sy-int(48*s)),(cx+int(16*s), sy)], fill=(50,50,80), width=6)
    # arms out
    d.line([(cx-int(14*s), sy-int(85*s)),(cx-int(45*s), sy-int(75*s))], fill=(255,200,150), width=5)
    d.line([(cx+int(14*s), sy-int(85*s)),(cx+int(45*s), sy-int(55*s))], fill=(255,200,150), width=5)
    # mirror
    d.rectangle([30, 80, 170, 220], fill=(210,240,255), outline=(160,130,90), width=4)
    draw_stick_figure(d, 100, 230, color=(220,60,60), scale=0.7)
    return img


def scene_go_to_school():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (180,220,255), (220,240,255))
    draw_ground(d, SCENE_SIZE, SCENE_SIZE, 300, color=(150,200,100))
    # path
    d.rectangle([150, 300, 250, SCENE_SIZE], fill=(220,200,160))
    draw_sun(d, 370, 60)
    draw_cloud(d, 100, 80)
    draw_cloud(d, 300, 50)
    # school building
    draw_house(d, 240, 140, w=200, h=160, wall=(255,240,210), roof=(200,60,60))
    # school sign
    d.rectangle([250, 168, 430, 195], fill=(255,255,255), outline=(100,100,100), width=2)
    # trees
    draw_tree(d, 60, 300, crown_r=40, color=(50,140,50))
    draw_tree(d, 420, 300, crown_r=35, color=(60,150,60))
    # Zoraiz walking with backpack
    cx, sy = 190, 390
    s = 0.9
    head_r = int(18*s)
    d.ellipse([cx-head_r, sy-int(100*s)-head_r*2, cx+head_r, sy-int(100*s)], fill=(255,200,150))
    d.rectangle([cx-int(14*s), sy-int(100*s), cx+int(14*s), sy-int(48*s)], fill=(220,60,60))
    d.line([(cx-int(8*s), sy-int(48*s)),(cx-int(20*s), sy)], fill=(50,50,80), width=6)
    d.line([(cx+int(8*s), sy-int(48*s)),(cx+int(10*s), sy)], fill=(50,50,80), width=6)
    # backpack on back
    d.rectangle([cx+12, sy-int(95*s), cx+30, sy-int(52*s)], fill=(60,100,200))
    # arm swinging
    d.line([(cx-int(14*s), sy-int(85*s)),(cx-int(38*s), sy-int(65*s))], fill=(255,200,150), width=5)
    d.line([(cx+int(14*s), sy-int(85*s)),(cx+int(30*s), sy-int(60*s))], fill=(255,200,150), width=5)
    return img


def scene_maestra():
    img, d = new_scene()
    d.rectangle([0, 0, SCENE_SIZE, SCENE_SIZE], fill=(245,250,220))
    d.rectangle([0, 330, SCENE_SIZE, SCENE_SIZE], fill=(200,180,150))
    # blackboard
    d.rectangle([50, 60, 380, 220], fill=(40,90,40), outline=(100,70,30), width=6)
    # writing on board
    for lx, ly in [(70,100),(70,130),(70,160),(70,190)]:
        d.line([(lx, ly),(lx+120+int(50*(ly%60/60)), ly)], fill=(255,255,200), width=3)
    # sun drawing on board
    d.ellipse([280, 90, 340, 150], fill=(255,220,0))
    # teacher (maestra Maria) at board
    draw_mama(d, 120, 340, scale=0.95)
    # pointer
    d.line([(145, 240),(200, 160)], fill=(140,100,60), width=4)
    # Zoraiz sitting at desk
    # desk
    d.rectangle([260, 270, 400, 290], fill=(200,160,100))
    d.rectangle([265, 290, 290, 340], fill=(180,140,80))
    d.rectangle([370, 290, 395, 340], fill=(180,140,80))
    # Zoraiz
    s = 0.8
    cx, sy = 320, 335
    head_r = int(18*s)
    d.ellipse([cx-head_r, sy-int(100*s)-head_r*2, cx+head_r, sy-int(100*s)], fill=(255,200,150))
    d.rectangle([cx-int(14*s), sy-int(100*s), cx+int(14*s), sy-int(48*s)], fill=(80,160,220))
    # book on desk
    d.rectangle([275, 255, 365, 272], fill=(220,60,60))
    return img


def scene_study():
    img, d = new_scene()
    d.rectangle([0, 0, SCENE_SIZE, SCENE_SIZE], fill=(240,248,255))
    d.rectangle([0, 330, SCENE_SIZE, SCENE_SIZE], fill=(210,190,160))
    # 3 desks with kids
    for i, (cx, color) in enumerate([(100,(80,160,220)),(210,(220,80,160)),(320,(80,200,120))]):
        # desk
        d.rectangle([cx-55, 270, cx+55, 290], fill=(200,160,100))
        d.rectangle([cx-50, 290, cx-30, 340], fill=(180,140,80))
        d.rectangle([cx+30, 290, cx+50, 340], fill=(180,140,80))
        # book
        d.rectangle([cx-40, 256, cx+40, 272], fill=(220,60,60) if i==0 else (60,100,200) if i==1 else (220,180,60))
        # kid
        s = 0.75
        sy = 340
        head_r = int(18*s)
        d.ellipse([cx-head_r, sy-int(100*s)-head_r*2, cx+head_r, sy-int(100*s)], fill=(255,200,150))
        d.rectangle([cx-int(14*s), sy-int(100*s), cx+int(14*s), sy-int(48*s)], fill=color)
        # arm writing
        d.line([(cx+int(14*s), sy-int(85*s)),(cx+int(45*s), sy-int(58*s))], fill=(255,200,150), width=4)
    # window
    d.rectangle([340, 40, 410, 140], fill=(180,230,255), outline=(160,140,100), width=3)
    draw_sun(d, 380, 20, r=25)
    return img


def scene_play_school():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (180,220,255), (220,240,255))
    draw_ground(d, SCENE_SIZE, SCENE_SIZE, 310, color=(140,200,80))
    draw_sun(d, 60, 60, r=35)
    # slide
    d.rectangle([280, 170, 340, 330], fill=(200,180,150))
    d.polygon([(200,330),(340,170),(340,200),(200,360)], fill=(100,180,220))
    # ladder
    for ly in [195, 230, 265, 300]:
        d.line([(328, ly),(328, ly+25)], fill=(140,120,90), width=4)
    # swing frame
    d.line([(60, 130),(60, 310)], fill=(140,120,90), width=5)
    d.line([(140, 130),(140, 310)], fill=(140,120,90), width=5)
    d.line([(60, 130),(140, 130)], fill=(140,120,90), width=5)
    d.line([(80, 130),(80, 230)], fill=(160,140,100), width=3)
    d.line([(120, 130),(120, 230)], fill=(160,140,100), width=3)
    d.rectangle([75, 225, 125, 240], fill=(80,120,200))
    # Zoraiz on swing
    s = 0.7
    cx, sy = 100, 310
    head_r = int(18*s)
    d.ellipse([cx-head_r, sy-int(100*s)-head_r*2, cx+head_r, sy-int(100*s)], fill=(255,200,150))
    d.rectangle([cx-int(14*s), sy-int(100*s), cx+int(14*s), sy-int(55*s)], fill=(220,60,60))
    d.line([(cx-int(14*s), sy-int(80*s)),(cx-int(40*s), sy-int(75*s))], fill=(255,200,150), width=4)
    d.line([(cx+int(14*s), sy-int(80*s)),(cx+int(40*s), sy-int(75*s))], fill=(255,200,150), width=4)
    # friend running
    draw_stick_figure(d, 230, 395, color=(80,200,120), scale=0.75)
    return img


def scene_come_home():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (200,230,255), (230,240,255))
    draw_ground(d, SCENE_SIZE, SCENE_SIZE, 295, color=(140,200,80))
    draw_sun(d, 370, 70, r=30)
    # path
    d.rectangle([100, 295, 240, SCENE_SIZE], fill=(220,200,160))
    # home
    draw_house(d, 30, 130, w=200, h=165, wall=(255,240,210), roof=(180,80,60))
    draw_tree(d, 330, 295, crown_r=40)
    # Zoraiz walking home (facing left)
    cx, sy = 280, 395
    s = 0.9
    head_r = int(18*s)
    d.ellipse([cx-head_r, sy-int(100*s)-head_r*2, cx+head_r, sy-int(100*s)], fill=(255,200,150))
    d.rectangle([cx-int(14*s), sy-int(100*s), cx+int(14*s), sy-int(48*s)], fill=(220,60,60))
    d.line([(cx-int(8*s), sy-int(48*s)),(cx-int(20*s), sy)], fill=(50,50,80), width=6)
    d.line([(cx+int(8*s), sy-int(48*s)),(cx+int(10*s), sy)], fill=(50,50,80), width=6)
    d.line([(cx-int(14*s), sy-int(85*s)),(cx-int(40*s), sy-int(60*s))], fill=(255,200,150), width=5)
    d.line([(cx+int(14*s), sy-int(85*s)),(cx+int(28*s), sy-int(65*s))], fill=(255,200,150), width=5)
    # backpack
    d.rectangle([cx-30, sy-int(95*s), cx-14, sy-int(52*s)], fill=(60,100,200))
    return img


def scene_lunch():
    img, d = new_scene()
    d.rectangle([0, 0, SCENE_SIZE, SCENE_SIZE], fill=(255,248,230))
    d.rectangle([0, 310, SCENE_SIZE, SCENE_SIZE], fill=(210,180,140))
    # table
    d.rectangle([40, 220, 390, 310], fill=(180,120,60))
    d.rectangle([50, 310, 100, 390], fill=(140,90,40))
    d.rectangle([330, 310, 380, 390], fill=(140,90,40))
    # big plate
    d.ellipse([110, 190, 310, 280], fill=(255,255,255), outline=(200,200,200), width=3)
    # pasta / food
    for i in range(8):
        import math
        angle = math.radians(i*45)
        fx = 210 + int(40*math.cos(angle))
        fy = 235 + int(30*math.sin(angle))
        d.arc([fx-20, fy-12, fx+20, fy+12], int(angle*57), int(angle*57)+180, fill=(220,160,60), width=5)
    # sauce red
    d.ellipse([185, 215, 235, 255], fill=(220,60,40,150))
    # glass
    d.rectangle([340, 195, 385, 250], fill=(200,230,255), outline=(150,180,220), width=2)
    # Zoraiz at table
    draw_stick_figure(d, 210, 410, color=(80,160,220), scale=0.85)
    # window
    d.rectangle([10, 60, 110, 170], fill=(180,230,255), outline=(160,140,100), width=3)
    return img


def scene_nap():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (180,200,240), (210,220,250))
    d.rectangle([0, 200, SCENE_SIZE, SCENE_SIZE], fill=(230,220,210))
    # bed
    d.rectangle([30, 180, 390, 340], fill=(160,100,60))
    d.rectangle([40, 130, 380, 320], fill=(255,255,255))
    d.rectangle([40, 130, 380, 195], fill=(120,160,220))  # pillow
    d.rectangle([40, 250, 380, 320], fill=(200,120,160))  # blanket
    # Zoraiz sleeping (head on pillow)
    d.ellipse([170, 145, 230, 205], fill=(255,200,150))
    # Zzz
    d.text((250, 110), "z z z", fill=(150,150,200))
    # moon in window
    d.rectangle([310, 40, 400, 130], fill=(30,30,80), outline=(160,140,100), width=3)
    draw_moon(d, 355, 85, r=25)
    draw_stars(d, 5)
    # curtain
    d.polygon([(30,0),(30,200),(80,150),(80,0)], fill=(200,100,150))
    d.polygon([(SCENE_SIZE-30,0),(SCENE_SIZE-30,200),(SCENE_SIZE-80,150),(SCENE_SIZE-80,0)], fill=(200,100,150))
    return img


def scene_park_afternoon():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (150,210,255), (200,230,255))
    draw_ground(d, SCENE_SIZE, SCENE_SIZE, 290, color=(120,190,70))
    draw_sun(d, 370, 55, r=38, color=(255,210,40))
    draw_cloud(d, 120, 80)
    draw_cloud(d, 310, 60)
    # path
    d.ellipse([80, 300, 350, 380], fill=(210,195,155), outline=(190,175,135), width=2)
    # trees
    draw_tree(d, 50, 295, trunk_h=50, crown_r=50, color=(40,140,40))
    draw_tree(d, 390, 290, trunk_h=45, crown_r=45, color=(50,150,50))
    draw_tree(d, 200, 285, trunk_h=35, crown_r=35, color=(60,160,60))
    # bench
    d.rectangle([240, 265, 370, 280], fill=(160,110,60))
    d.rectangle([248, 280, 268, 310], fill=(130,90,50))
    d.rectangle([345, 280, 365, 310], fill=(130,90,50))
    # Zoraiz running to park
    cx, sy = 150, 390
    s = 1.0
    head_r = int(18*s)
    d.ellipse([cx-head_r, sy-int(100*s)-head_r*2, cx+head_r, sy-int(100*s)], fill=(255,200,150))
    d.rectangle([cx-int(14*s), sy-int(100*s), cx+int(14*s), sy-int(48*s)], fill=(80,180,100))
    d.line([(cx-int(8*s), sy-int(48*s)),(cx-int(25*s), sy)], fill=(50,50,80), width=6)
    d.line([(cx+int(8*s), sy-int(48*s)),(cx+int(5*s), sy)], fill=(50,50,80), width=6)
    d.line([(cx-int(14*s), sy-int(85*s)),(cx-int(42*s), sy-int(55*s))], fill=(255,200,150), width=5)
    d.line([(cx+int(14*s), sy-int(85*s)),(cx+int(38*s), sy-int(70*s))], fill=(255,200,150), width=5)
    return img


def scene_play_park():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (160,215,255), (200,235,255))
    draw_ground(d, SCENE_SIZE, SCENE_SIZE, 300, color=(120,190,70))
    draw_sun(d, 60, 70, r=35)
    # ball
    d.ellipse([185, 260, 240, 315], fill=(220,60,60))
    d.arc([185, 260, 240, 315], 30, 150, fill=(255,255,255), width=3)
    # 3 kids playing
    for i, (cx, color) in enumerate([(100,(80,160,220)),(210,(220,80,160)),(320,(80,200,120))]):
        s = 0.85
        sy = 400
        head_r = int(18*s)
        d.ellipse([cx-head_r, sy-int(100*s)-head_r*2, cx+head_r, sy-int(100*s)], fill=(255,200,150))
        d.rectangle([cx-int(14*s), sy-int(100*s), cx+int(14*s), sy-int(48*s)], fill=color)
        d.line([(cx-int(8*s), sy-int(48*s)),(cx-int(16*s), sy)], fill=(80,80,80), width=5)
        d.line([(cx+int(8*s), sy-int(48*s)),(cx+int(16*s), sy)], fill=(80,80,80), width=5)
        if i == 0:
            d.line([(cx+int(14*s), sy-int(85*s)),(cx+int(50*s), sy-int(60*s))], fill=(255,200,150), width=5)
        elif i == 2:
            d.line([(cx-int(14*s), sy-int(85*s)),(cx-int(50*s), sy-int(60*s))], fill=(255,200,150), width=5)
    # trees
    draw_tree(d, 390, 300, crown_r=40)
    draw_tree(d, 30, 300, crown_r=38)
    return img


def scene_dinner():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (80,60,130), (120,90,160))
    d.rectangle([0, 200, SCENE_SIZE, SCENE_SIZE], fill=(210,180,140))
    d.rectangle([0, 70, SCENE_SIZE, 200], fill=(255,248,235))
    # window with moon
    d.rectangle([300, 80, 400, 170], fill=(30,30,80), outline=(160,140,100), width=3)
    draw_moon(d, 350, 125, r=22)
    # lamp
    d.polygon([(195, 40),(225, 40),(240, 90),(180, 90)], fill=(255,220,100))
    d.line([(210, 40),(210, 0)], fill=(180,150,80), width=4)
    d.ellipse([185, 85, 235, 115], fill=(255,240,150))
    # table
    d.rectangle([30, 210, 400, 300], fill=(160,110,60))
    d.rectangle([40, 300, 90, 400], fill=(130,90,50))
    d.rectangle([330, 300, 380, 400], fill=(130,90,50))
    # plates
    for px in [90, 205, 315]:
        d.ellipse([px-45, 188, px+45, 255], fill=(255,255,255), outline=(200,200,200), width=2)
        d.ellipse([px-30, 198, px+30, 245], fill=(240,200,140))
    # candle centerpiece
    d.rectangle([200, 175, 215, 215], fill=(255,240,200))
    d.ellipse([197, 168, 218, 182], fill=(255,180,50))
    # Zoraiz center
    draw_stick_figure(d, 210, 420, color=(80,160,220), scale=0.75)
    draw_mama(d, 80, 420, scale=0.75)
    draw_baba(d, 350, 420, scale=0.75)
    return img


def scene_good_night():
    img, d = new_scene()
    draw_sky(d, SCENE_SIZE, SCENE_SIZE, (20,20,70), (60,40,100))
    draw_stars(d, 15)
    draw_moon(d, 340, 80, r=45)
    d.rectangle([0, 240, SCENE_SIZE, SCENE_SIZE], fill=(210,200,190))
    # bed
    d.rectangle([30, 220, 390, 370], fill=(140,90,60))
    d.rectangle([40, 175, 380, 350], fill=(255,255,255))
    d.rectangle([40, 175, 380, 230], fill=(100,130,200))   # pillow
    d.rectangle([40, 280, 380, 355], fill=(150,100,160))   # blanket with stars pattern
    for sx in range(70, 370, 40):
        for sy in range(295, 355, 25):
            d.polygon([(sx,sy-6),(sx+3,sy-2),(sx+8,sy-2),(sx+4,sy+2),(sx+6,sy+7),
                       (sx,sy+4),(sx-6,sy+7),(sx-4,sy+2),(sx-8,sy-2),(sx-3,sy-2)],
                      fill=(200,160,220))
    # Zoraiz sleeping head on pillow - eyes closed
    d.ellipse([165, 183, 225, 240], fill=(255,200,150))
    # closed eyes
    d.arc([175, 198, 193, 210], 0, 180, fill=(100,70,50), width=3)
    d.arc([198, 198, 216, 210], 0, 180, fill=(100,70,50), width=3)
    # small smile
    d.arc([183, 215, 208, 228], 0, 180, fill=(200,120,100), width=2)
    # Zzz bubbles
    for i, (zx, zy, sz) in enumerate([(240, 165, 18),(265, 140, 22),(296, 112, 28)]):
        d.text((zx, zy), "Z", fill=(180,180,220))
    # teddy bear
    d.ellipse([320, 270, 380, 340], fill=(200,150,100))
    d.ellipse([310, 255, 345, 285], fill=(200,150,100))
    d.ellipse([352, 255, 385, 280], fill=(200,150,100))
    d.ellipse([313, 258, 328, 273], fill=(180,130,80))
    d.ellipse([358, 258, 373, 273], fill=(180,130,80))
    d.ellipse([322, 278, 375, 335], fill=(200,150,100))
    return img


# ── Page data ──────────────────────────────────────────────────────────────────

PAGES = [
    dict(scene_fn=scene_wake_up,
         line1="Zoraiz si sveglia alle sette!",
         line2="Buongiorno Zoraiz!",
         bg="#FFF9C4", acc="#F9A825"),
    dict(scene_fn=scene_hug_mama,
         line1="Zoraiz corre dalla mama.",
         line2="Un abbraccio, mama!",
         bg="#FCE4EC", acc="#E91E8C"),
    dict(scene_fn=scene_bathroom,
         line1="Zoraiz va in bagno.",
         line2="Si lava i denti, le mani e la faccia!",
         bg="#E0F7FA", acc="#00838F"),
    dict(scene_fn=scene_breakfast,
         line1="Colazione con mama e baba!",
         line2="Latte al cioccolato, uova e pane.",
         bg="#FFF3E0", acc="#E65100"),
    dict(scene_fn=scene_get_ready,
         line1="Zoraiz si veste per la scuola.",
         line2="Zaino in spalla, si parte!",
         bg="#E8F5E9", acc="#2E7D32"),
    dict(scene_fn=scene_go_to_school,
         line1="Zoraiz va a scuola.",
         line2="Cammina felice!",
         bg="#E3F2FD", acc="#1565C0"),
    dict(scene_fn=scene_maestra,
         line1="Ciao maestra Maria!",
         line2="La maestra è molto brava.",
         bg="#F3E5F5", acc="#6A1B9A"),
    dict(scene_fn=scene_study,
         line1="Zoraiz studia con i suoi amici.",
         line2="Bravo Zoraiz!",
         bg="#FFFDE7", acc="#F57F17"),
    dict(scene_fn=scene_play_school,
         line1="Zoraiz gioca con gli amici.",
         line2="Che divertimento a scuola!",
         bg="#E8F5E9", acc="#388E3C"),
    dict(scene_fn=scene_come_home,
         line1="Zoraiz torna a casa.",
         line2="Arrivederci amici!",
         bg="#FFF8E1", acc="#FF8F00"),
    dict(scene_fn=scene_lunch,
         line1="Zoraiz mangia il pranzo.",
         line2="Che buono!",
         bg="#FBE9E7", acc="#BF360C"),
    dict(scene_fn=scene_nap,
         line1="Zoraiz fa il pisolino.",
         line2="Sogni d'oro, Zoraiz!",
         bg="#EDE7F6", acc="#512DA8"),
    dict(scene_fn=scene_park_afternoon,
         line1="Nel pomeriggio Zoraiz va al parco.",
         line2="Che bella giornata!",
         bg="#E8F5E9", acc="#2E7D32"),
    dict(scene_fn=scene_play_park,
         line1="Zoraiz gioca con i bambini.",
         line2="Che bello giocare insieme!",
         bg="#E3F2FD", acc="#1565C0"),
    dict(scene_fn=scene_dinner,
         line1="Zoraiz cena con mama e baba.",
         line2="La famiglia è insieme!",
         bg="#F9FBE7", acc="#558B2F"),
    dict(scene_fn=scene_good_night,
         line1="Zoraiz dorme.",
         line2="Buonanotte Zoraiz! Ti vogliamo bene!",
         bg="#1A237E", acc="#7986CB"),
]


# ── PDF drawing ────────────────────────────────────────────────────────────────

def draw_cover(c):
    # gradient background
    c.setFillColor(HexColor("#FF8F00"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(HexColor("#FCE4EC"))
    c.roundRect(0, 0, W, H*0.42, 0, fill=1, stroke=0)

    # top banner
    c.setFillColor(HexColor("#FF8F00"))
    c.roundRect(18, H-155, W-36, 140, 24, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 44)
    c.drawCentredString(W/2, H-80, "La Giornata di Zoraiz")
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(W/2, H-120, "Impariamo l'italiano!")

    # central illustration
    try:
        scene_img = scene_wake_up()
        scene_img = scene_img.resize((360, 360))
        # rounded corners
        mask = Image.new("L", scene_img.size, 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle([0,0,359,359], radius=60, fill=255)
        scene_img = scene_img.convert("RGBA")
        scene_img.putalpha(mask)
        px = (W-360)/2
        py = H-155-360-20
        c.setFillColor(HexColor("#BDBDBD"))
        c.roundRect(px+8, py-8, 360, 360, 22, fill=1, stroke=0)
        c.setStrokeColor(HexColor("#FF8F00"))
        c.setLineWidth(6)
        c.setFillColor(white)
        c.roundRect(px-10, py-10, 380, 380, 26, fill=1, stroke=1)
        c.drawImage(to_reader(scene_img), px, py, width=360, height=360, mask='auto')
    except Exception as e:
        print(f"Cover error: {e}")

    # bottom box
    c.setFillColor(HexColor("#FF8F00"))
    c.roundRect(22, 32, W-44, 96, 18, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(W/2, 100, "Dal mattino alla sera con Zoraiz!")
    c.setFont("Helvetica", 16)
    c.drawCentredString(W/2, 68, f"{len(PAGES)} pagine di avventure")
    for hx in [55, W-55]:
        c.setFont("Helvetica-Bold", 22)
        c.drawCentredString(hx, 80, "♥")


def draw_page(c, page, num, total):
    bg = hex_rgb(page["bg"])
    acc = hex_rgb(page["acc"])

    c.setFillColorRGB(*bg)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # top bar
    c.setFillColorRGB(*acc)
    c.rect(0, H-56, W, 56, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(W/2, H-36, "La Giornata di Zoraiz")

    # page number
    c.setFillColorRGB(*bg)
    c.circle(W-36, H-28, 18, fill=1, stroke=0)
    c.setFillColorRGB(*acc)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(W-36, H-33, str(num))

    # illustration
    try:
        scene_img = page["scene_fn"]()
        img_size = 400
        scene_img = scene_img.resize((img_size, img_size), Image.LANCZOS)
        # rounded corners
        mask = Image.new("L", (img_size, img_size), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle([0,0,img_size-1,img_size-1], radius=70, fill=255)
        scene_img = scene_img.convert("RGBA")
        scene_img.putalpha(mask)

        ix = (W - img_size) / 2
        iy = H - 62 - img_size - 8

        c.setFillColorRGB(0.72, 0.72, 0.72)
        c.roundRect(ix+7, iy-7, img_size, img_size, 20, fill=1, stroke=0)
        c.setFillColor(white)
        c.setStrokeColorRGB(*acc)
        c.setLineWidth(5)
        c.roundRect(ix-10, iy-10, img_size+20, img_size+20, 24, fill=1, stroke=1)
        c.drawImage(to_reader(scene_img), ix, iy, width=img_size, height=img_size, mask='auto')
    except Exception as e:
        print(f"Page {num} scene error: {e}")
        iy = H - 62 - 400 - 8

    # sentence box
    bx = 22; by = iy - 126; bw = W - 44; bh = 114
    c.setFillColorRGB(*acc)
    c.roundRect(bx, by, bw, bh, 16, fill=1, stroke=0)
    c.setFillColor(white)
    line1 = page["line1"]
    line2 = page.get("line2", "")
    if line2:
        c.setFont("Helvetica-Bold", 23)
        c.drawCentredString(W/2, by + bh - 36, line1)
        c.setFont("Helvetica-Bold", 21)
        c.drawCentredString(W/2, by + bh - 68, line2)
    else:
        c.setFont("Helvetica-Bold", 26)
        c.drawCentredString(W/2, by + bh - 48, line1)

    # progress dots
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
