import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arc, FancyArrowPatch
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
import io
import os

W, H = A4  # 595 x 842 pts

PAGES = [
    {
        "title": "Zoraiz è il bambino.",
        "subtitle": "Mamma e Babbo lo amano tanto!",
        "scene": "cover",
        "bg": "#FFF9C4",
    },
    {
        "title": "Zoraiz mangia il gelato.",
        "subtitle": "Che buono! 🍦",
        "scene": "icecream",
        "bg": "#E3F2FD",
    },
    {
        "title": "Zoraiz e Babbo dormono.",
        "subtitle": "Buona notte! 🌙",
        "scene": "sleeping",
        "bg": "#EDE7F6",
    },
    {
        "title": "Zoraiz fa il bagno.",
        "subtitle": "Splash splash! 🛁",
        "scene": "bath",
        "bg": "#E0F7FA",
    },
    {
        "title": "Zoraiz gioca con i giocattoli.",
        "subtitle": "Che divertimento! 🎈",
        "scene": "toys",
        "bg": "#FCE4EC",
    },
    {
        "title": "Zoraiz e Mamma cucinano.",
        "subtitle": "Mmmm, che profumo! 🍳",
        "scene": "cooking",
        "bg": "#FFF3E0",
    },
    {
        "title": "Zoraiz corre nel parco.",
        "subtitle": "Vai Zoraiz, vai! 🌳",
        "scene": "park",
        "bg": "#F1F8E9",
    },
    {
        "title": "Zoraiz legge un libro.",
        "subtitle": "Bravo Zoraiz! 📚",
        "scene": "reading",
        "bg": "#FBE9E7",
    },
    {
        "title": "Zoraiz e Mamma ballano.",
        "subtitle": "Che bella musica! 🎵",
        "scene": "dancing",
        "bg": "#F3E5F5",
    },
    {
        "title": "Zoraiz abbraccia Babbo.",
        "subtitle": "Ti voglio bene, Babbo! ❤️",
        "scene": "hug",
        "bg": "#FFEBEE",
    },
]


def baby_face(ax, cx, cy, r=0.12, skin="#FDDBB4", hat_color=None):
    """Draw a cute baby face."""
    face = Circle((cx, cy), r, color=skin, zorder=5)
    ax.add_patch(face)
    # eyes
    ax.add_patch(Circle((cx - r*0.3, cy + r*0.1), r*0.1, color="#333", zorder=6))
    ax.add_patch(Circle((cx + r*0.3, cy + r*0.1), r*0.1, color="#333", zorder=6))
    # smile
    smile = Arc((cx, cy - r*0.05), r*0.4, r*0.25, angle=0, theta1=200, theta2=340,
                color="#c0392b", lw=2, zorder=6)
    ax.add_patch(smile)
    # hair
    hair = patches.Wedge((cx, cy + r*0.5), r*0.6, 210, 330, color="#6D4C41", zorder=4)
    ax.add_patch(hair)
    if hat_color:
        hat = patches.Wedge((cx, cy + r*0.4), r*0.7, 180, 360, color=hat_color, zorder=7)
        ax.add_patch(hat)


def adult_face(ax, cx, cy, r=0.13, skin="#FDDBB4", is_mama=False):
    face = Circle((cx, cy), r, color=skin, zorder=5)
    ax.add_patch(face)
    ax.add_patch(Circle((cx - r*0.3, cy + r*0.1), r*0.1, color="#333", zorder=6))
    ax.add_patch(Circle((cx + r*0.3, cy + r*0.1), r*0.1, color="#333", zorder=6))
    smile = Arc((cx, cy - r*0.05), r*0.4, r*0.25, angle=0, theta1=200, theta2=340,
                color="#c0392b", lw=2, zorder=6)
    ax.add_patch(smile)
    if is_mama:
        # long hair
        for dx in [-1, 1]:
            hair = patches.FancyBboxPatch((cx + dx*r*0.7, cy - r*1.2), r*0.4, r*1.8,
                                          boxstyle="round,pad=0.02", color="#4E342E", zorder=4)
            ax.add_patch(hair)
        top = patches.Wedge((cx, cy + r*0.5), r*0.75, 180, 360, color="#4E342E", zorder=4)
        ax.add_patch(top)
    else:
        top = patches.Wedge((cx, cy + r*0.5), r*0.65, 180, 360, color="#37474F", zorder=4)
        ax.add_patch(top)


def draw_ground(ax, color="#A5D6A7", y=0.18):
    ground = FancyBboxPatch((0, 0), 1, y, boxstyle="square", color=color, zorder=1)
    ax.add_patch(ground)


def draw_sky(ax, color="#BBDEFB", ystart=0.5):
    sky = FancyBboxPatch((0, ystart), 1, 1 - ystart, boxstyle="square", color=color, zorder=0)
    ax.add_patch(sky)


def draw_sun(ax, cx=0.85, cy=0.85, r=0.07):
    ax.add_patch(Circle((cx, cy), r, color="#FDD835", zorder=2))
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        x1, y1 = cx + r*1.1*np.cos(rad), cy + r*1.1*np.sin(rad)
        x2, y2 = cx + r*1.5*np.cos(rad), cy + r*1.5*np.sin(rad)
        ax.plot([x1, x2], [y1, y2], color="#FDD835", lw=2, zorder=2)


def scene_cover(ax):
    ax.set_facecolor("#FFF9C4")
    draw_sky(ax, "#B3E5FC", 0.55)
    draw_ground(ax, "#C8E6C9", 0.35)
    draw_sun(ax)
    # rainbow
    for i, c in enumerate(["#E53935","#FB8C00","#FDD835","#43A047","#1E88E5","#8E24AA"]):
        r = 0.55 - i*0.04
        arc = Arc((0.5, 0.35), r*2, r*2, angle=0, theta1=0, theta2=180,
                  color=c, lw=4, zorder=2, alpha=0.8)
        ax.add_patch(arc)
    # baby in center
    baby_face(ax, 0.5, 0.52, r=0.14)
    # body
    body = FancyBboxPatch((0.38, 0.30), 0.24, 0.22, boxstyle="round,pad=0.02",
                           color="#EF9A9A", zorder=4)
    ax.add_patch(body)
    # mama left
    adult_face(ax, 0.22, 0.54, r=0.11, is_mama=True)
    body_m = FancyBboxPatch((0.12, 0.33), 0.20, 0.21, boxstyle="round,pad=0.02",
                             color="#F48FB1", zorder=4)
    ax.add_patch(body_m)
    # baba right
    adult_face(ax, 0.78, 0.54, r=0.11, is_mama=False)
    body_b = FancyBboxPatch((0.68, 0.33), 0.20, 0.21, boxstyle="round,pad=0.02",
                             color="#90CAF9", zorder=4)
    ax.add_patch(body_b)
    # hearts
    for (hx, hy) in [(0.35, 0.68), (0.65, 0.68)]:
        ax.text(hx, hy, "❤", fontsize=18, ha='center', va='center', zorder=8)
    # flowers
    for fx in [0.1, 0.2, 0.8, 0.9]:
        stem = patches.FancyArrowPatch((fx, 0.18), (fx, 0.32), color="#388E3C",
                                       arrowstyle="-", lw=2, zorder=3)
        ax.add_patch(stem)
        ax.add_patch(Circle((fx, 0.32), 0.03, color="#E91E63", zorder=3))


def scene_icecream(ax):
    ax.set_facecolor("#E3F2FD")
    draw_sky(ax, "#B3E5FC", 0.5)
    draw_ground(ax, "#FFF9C4", 0.28)
    draw_sun(ax)
    # ice cream shop sign
    sign = FancyBboxPatch((0.62, 0.55), 0.30, 0.15, boxstyle="round,pad=0.01",
                           color="#F8BBD9", zorder=3)
    ax.add_patch(sign)
    ax.text(0.77, 0.625, "Gelato! 🍦", fontsize=8, ha='center', va='center',
            fontweight='bold', color="#880E4F", zorder=4)
    # table
    table = FancyBboxPatch((0.55, 0.30), 0.35, 0.06, boxstyle="round,pad=0.01",
                            color="#BCAAA4", zorder=3)
    ax.add_patch(table)
    # ice cream cone
    cone_x, cone_y = 0.72, 0.36
    triangle = plt.Polygon([[cone_x, cone_y], [cone_x-0.05, cone_y+0.12],
                             [cone_x+0.05, cone_y+0.12]], color="#D7CCC8", zorder=5)
    ax.add_patch(triangle)
    ax.add_patch(Circle((cone_x, cone_y+0.14), 0.055, color="#F48FB1", zorder=6))
    ax.add_patch(Circle((cone_x, cone_y+0.17), 0.04, color="#FFCC80", zorder=7))
    ax.text(cone_x+0.005, cone_y+0.215, "🍓", fontsize=12, ha='center', zorder=8)
    # baby sitting at table
    baby_face(ax, 0.62, 0.50, r=0.10)
    body = FancyBboxPatch((0.53, 0.30), 0.18, 0.20, boxstyle="round,pad=0.02",
                           color="#80DEEA", zorder=4)
    ax.add_patch(body)
    # arm reaching
    ax.plot([0.62, 0.69], [0.40, 0.40], color="#FDDBB4", lw=6, zorder=5)
    # drip drops
    for (dx, dy) in [(0.70, 0.52), (0.68, 0.48)]:
        ax.add_patch(Circle((dx, dy), 0.012, color="#F48FB1", zorder=6, alpha=0.7))


def scene_sleeping(ax):
    ax.set_facecolor("#EDE7F6")
    # night sky
    sky = FancyBboxPatch((0, 0.4), 1, 0.6, boxstyle="square", color="#1A237E", zorder=1)
    ax.add_patch(sky)
    # moon
    ax.add_patch(Circle((0.82, 0.80), 0.07, color="#FFF9C4", zorder=2))
    ax.add_patch(Circle((0.88, 0.82), 0.06, color="#1A237E", zorder=3))
    # stars
    for (sx, sy) in [(0.1,0.85),(0.25,0.92),(0.45,0.88),(0.6,0.95),(0.35,0.78)]:
        ax.text(sx, sy, "★", fontsize=10, color="#FFF9C4", ha='center', zorder=4)
    # floor
    floor = FancyBboxPatch((0, 0), 1, 0.4, boxstyle="square", color="#D7CCC8", zorder=1)
    ax.add_patch(floor)
    # big bed
    bed = FancyBboxPatch((0.08, 0.22), 0.84, 0.28, boxstyle="round,pad=0.02",
                          color="#8D6E63", zorder=2)
    ax.add_patch(bed)
    pillow_l = FancyBboxPatch((0.12, 0.38), 0.22, 0.09, boxstyle="round,pad=0.02",
                               color="#F8BBD9", zorder=3)
    pillow_r = FancyBboxPatch((0.65, 0.38), 0.22, 0.09, boxstyle="round,pad=0.02",
                               color="#BBDEFB", zorder=3)
    ax.add_patch(pillow_l)
    ax.add_patch(pillow_r)
    # blanket
    blanket = FancyBboxPatch((0.10, 0.22), 0.80, 0.18, boxstyle="round,pad=0.02",
                              color="#CE93D8", zorder=3)
    ax.add_patch(blanket)
    # star pattern on blanket
    for bx in [0.2, 0.35, 0.5, 0.65, 0.8]:
        ax.text(bx, 0.30, "⭐", fontsize=8, ha='center', zorder=4, alpha=0.7)
    # baby head left
    baby_face(ax, 0.23, 0.46, r=0.09)
    ax.text(0.23, 0.46, "z", fontsize=9, color="#7B1FA2", ha='center', va='bottom',
            fontstyle='italic', zorder=8)
    # baba head right - with closed eyes
    adult_face(ax, 0.77, 0.46, r=0.11)
    ax.plot([0.73, 0.76], [0.47, 0.47], color="#333", lw=2, zorder=8)
    ax.plot([0.78, 0.81], [0.47, 0.47], color="#333", lw=2, zorder=8)
    ax.text(0.85, 0.55, "z z z", fontsize=11, color="#7B1FA2", ha='center', zorder=8,
            fontstyle='italic')


def scene_bath(ax):
    ax.set_facecolor("#E0F7FA")
    # wall tiles
    for tx in range(0, 10):
        for ty in range(5, 10):
            tile = FancyBboxPatch((tx*0.1, ty*0.1), 0.09, 0.09,
                                   boxstyle="square", color="#B2EBF2", zorder=1, lw=0.5,
                                   ec="#80DEEA")
            ax.add_patch(tile)
    # bathtub
    tub = FancyBboxPatch((0.10, 0.12), 0.80, 0.35, boxstyle="round,pad=0.03",
                          color="#E0E0E0", zorder=2)
    ax.add_patch(tub)
    # water
    water = FancyBboxPatch((0.13, 0.14), 0.74, 0.20, boxstyle="round,pad=0.02",
                            color="#4DD0E1", zorder=3, alpha=0.7)
    ax.add_patch(water)
    # bubbles
    for (bx, by, br) in [(0.25,0.38,0.04),(0.45,0.42,0.05),(0.60,0.37,0.03),
                          (0.70,0.43,0.04),(0.35,0.44,0.035)]:
        ax.add_patch(Circle((bx, by), br, color="white", zorder=5, alpha=0.8))
        ax.add_patch(Circle((bx+br*0.3, by+br*0.3), br*0.3, color="#B2EBF2",
                             zorder=6, alpha=0.6))
    # baby in tub
    baby_face(ax, 0.50, 0.40, r=0.11)
    # rubber duck
    ax.add_patch(Circle((0.78, 0.30), 0.04, color="#FDD835", zorder=5))
    ax.add_patch(Circle((0.80, 0.33), 0.025, color="#FDD835", zorder=6))
    ax.add_patch(patches.Wedge((0.81, 0.335), 0.015, 0, 30, color="#FF8F00", zorder=7))
    # splash drops
    for (sx, sy) in [(0.30,0.50),(0.55,0.53),(0.40,0.52)]:
        ax.text(sx, sy, "💧", fontsize=12, ha='center', zorder=6)
    # faucet
    faucet = FancyBboxPatch((0.44, 0.47), 0.12, 0.04, boxstyle="round,pad=0.01",
                             color="#9E9E9E", zorder=4)
    ax.add_patch(faucet)


def scene_toys(ax):
    ax.set_facecolor("#FCE4EC")
    draw_sky(ax, "#FCE4EC", 0.5)
    draw_ground(ax, "#F8BBD9", 0.22)
    # colourful blocks
    for i, (bx, col) in enumerate([(0.10,"#EF5350"),(0.22,"#FFA726"),
                                     (0.34,"#FFEE58"),(0.46,"#66BB6A")]):
        block = FancyBboxPatch((bx, 0.22), 0.10, 0.12, boxstyle="round,pad=0.01",
                               color=col, zorder=3)
        ax.add_patch(block)
        ax.text(bx+0.05, 0.285, str(i+1), fontsize=12, ha='center', va='center',
                color="white", fontweight='bold', zorder=4)
    # balloon
    ax.add_patch(Circle((0.75, 0.70), 0.08, color="#EF5350", zorder=3))
    ax.plot([0.75, 0.72], [0.62, 0.40], color="#B71C1C", lw=1.5, zorder=3)
    ax.add_patch(Circle((0.55, 0.65), 0.07, color="#FFA726", zorder=3))
    ax.plot([0.55, 0.57], [0.58, 0.40], color="#E65100", lw=1.5, zorder=3)
    ax.add_patch(Circle((0.88, 0.62), 0.06, color="#AB47BC", zorder=3))
    ax.plot([0.88, 0.86], [0.56, 0.38], color="#6A1B9A", lw=1.5, zorder=3)
    # baby playing
    baby_face(ax, 0.50, 0.43, r=0.11, hat_color="#42A5F5")
    body = FancyBboxPatch((0.41, 0.22), 0.18, 0.21, boxstyle="round,pad=0.02",
                           color="#CE93D8", zorder=4)
    ax.add_patch(body)
    # arms up happy
    ax.plot([0.41, 0.28], [0.35, 0.44], color="#FDDBB4", lw=6, zorder=5)
    ax.plot([0.59, 0.72], [0.35, 0.44], color="#FDDBB4", lw=6, zorder=5)


def scene_cooking(ax):
    ax.set_facecolor("#FFF3E0")
    # kitchen wall
    wall = FancyBboxPatch((0, 0.45), 1, 0.55, boxstyle="square", color="#FFCCBC", zorder=1)
    ax.add_patch(wall)
    # floor
    floor = FancyBboxPatch((0, 0), 1, 0.45, boxstyle="square", color="#FFECB3", zorder=1)
    ax.add_patch(floor)
    # counter
    counter = FancyBboxPatch((0.05, 0.38), 0.90, 0.12, boxstyle="round,pad=0.01",
                              color="#8D6E63", zorder=2)
    ax.add_patch(counter)
    # stove pot
    pot = FancyBboxPatch((0.58, 0.42), 0.25, 0.20, boxstyle="round,pad=0.02",
                          color="#78909C", zorder=3)
    ax.add_patch(pot)
    # handle
    ax.plot([0.61, 0.56], [0.55, 0.55], color="#546E7A", lw=5, zorder=4)
    ax.plot([0.80, 0.85], [0.55, 0.55], color="#546E7A", lw=5, zorder=4)
    # steam
    for (sx, soff) in [(0.65,0),(0.72,0.03),(0.79,0)]:
        ax.plot([sx, sx+soff, sx], [0.62, 0.70, 0.78], color="#B0BEC5",
                lw=2, alpha=0.7, zorder=4)
    # window
    win = FancyBboxPatch((0.70, 0.60), 0.22, 0.25, boxstyle="round,pad=0.01",
                          color="#B3E5FC", zorder=2)
    ax.add_patch(win)
    ax.plot([0.81, 0.81], [0.60, 0.85], color="#90CAF9", lw=2, zorder=3)
    ax.plot([0.70, 0.92], [0.725, 0.725], color="#90CAF9", lw=2, zorder=3)
    # mama
    adult_face(ax, 0.35, 0.72, r=0.12, is_mama=True)
    body_m = FancyBboxPatch((0.25, 0.45), 0.20, 0.27, boxstyle="round,pad=0.02",
                             color="#F06292", zorder=4)
    ax.add_patch(body_m)
    # apron
    apron = FancyBboxPatch((0.27, 0.46), 0.16, 0.22, boxstyle="round,pad=0.01",
                            color="white", zorder=5, alpha=0.8)
    ax.add_patch(apron)
    # baby on stool
    stool = FancyBboxPatch((0.53, 0.28), 0.14, 0.12, boxstyle="round,pad=0.01",
                            color="#A1887F", zorder=3)
    ax.add_patch(stool)
    baby_face(ax, 0.60, 0.52, r=0.09)
    body_b = FancyBboxPatch((0.52, 0.32), 0.16, 0.17, boxstyle="round,pad=0.02",
                             color="#80CBC4", zorder=4)
    ax.add_patch(body_b)
    # spoon in baby's hand
    ax.plot([0.56, 0.50], [0.40, 0.48], color="#9E9E9E", lw=4, zorder=6)
    ax.add_patch(Circle((0.49, 0.49), 0.025, color="#9E9E9E", zorder=7))


def scene_park(ax):
    ax.set_facecolor("#F1F8E9")
    draw_sky(ax, "#B3E5FC", 0.45)
    draw_ground(ax, "#8BC34A", 0.28)
    draw_sun(ax, 0.15, 0.85)
    # clouds
    for (cx, cy) in [(0.35, 0.80), (0.65, 0.75)]:
        for (dx, dy, r) in [(0,0,0.07),(0.06,0.02,0.06),(-0.05,0.02,0.055)]:
            ax.add_patch(Circle((cx+dx, cy+dy), r, color="white", zorder=2, alpha=0.9))
    # trees
    for (tx, tc) in [(0.10, "#388E3C"), (0.85, "#2E7D32")]:
        trunk = FancyBboxPatch((tx-0.02, 0.20), 0.04, 0.15, boxstyle="square",
                               color="#795548", zorder=3)
        ax.add_patch(trunk)
        ax.add_patch(Circle((tx, 0.40), 0.10, color=tc, zorder=4))
        ax.add_patch(Circle((tx-0.07, 0.37), 0.07, color=tc, zorder=4))
        ax.add_patch(Circle((tx+0.07, 0.37), 0.07, color=tc, zorder=4))
    # path
    path = FancyBboxPatch((0.20, 0.18), 0.60, 0.10, boxstyle="round,pad=0.01",
                           color="#D7CCC8", zorder=2)
    ax.add_patch(path)
    # flowers on grass
    for (fx, fy, fc) in [(0.30,0.29,"#FF80AB"),(0.55,0.31,"#FFEB3B"),
                          (0.70,0.28,"#F06292"),(0.15,0.30,"#FF8F00")]:
        ax.add_patch(Circle((fx, fy), 0.025, color=fc, zorder=4))
        ax.plot([fx, fx], [fy-0.025, fy-0.07], color="#558B2F", lw=2, zorder=3)
    # baby running
    baby_face(ax, 0.50, 0.50, r=0.10, hat_color="#FF5722")
    body = FancyBboxPatch((0.42, 0.30), 0.16, 0.20, boxstyle="round,pad=0.02",
                           color="#4FC3F7", zorder=4)
    ax.add_patch(body)
    # running legs
    ax.plot([0.46, 0.38], [0.30, 0.20], color="#FDDBB4", lw=5, zorder=5)
    ax.plot([0.54, 0.63], [0.30, 0.20], color="#FDDBB4", lw=5, zorder=5)
    # arms swinging
    ax.plot([0.42, 0.32], [0.38, 0.44], color="#FDDBB4", lw=5, zorder=5)
    ax.plot([0.58, 0.68], [0.38, 0.30], color="#FDDBB4", lw=5, zorder=5)
    # butterfly
    ax.text(0.72, 0.55, "🦋", fontsize=16, zorder=6)


def scene_reading(ax):
    ax.set_facecolor("#FBE9E7")
    # cosy rug
    rug = FancyBboxPatch((0.10, 0.08), 0.80, 0.35, boxstyle="round,pad=0.03",
                          color="#EF9A9A", zorder=2)
    ax.add_patch(rug)
    # rug pattern
    inner = FancyBboxPatch((0.14, 0.11), 0.72, 0.29, boxstyle="round,pad=0.02",
                            color="#EF5350", zorder=3, alpha=0.4)
    ax.add_patch(inner)
    # bookshelf
    shelf = FancyBboxPatch((0.60, 0.40), 0.35, 0.55, boxstyle="round,pad=0.01",
                            color="#8D6E63", zorder=2)
    ax.add_patch(shelf)
    for sy in [0.55, 0.70, 0.85]:
        bar = FancyBboxPatch((0.61, sy), 0.33, 0.02, boxstyle="square",
                              color="#6D4C41", zorder=3)
        ax.add_patch(bar)
    # books on shelf
    book_colors = ["#EF5350","#42A5F5","#66BB6A","#FFA726","#AB47BC","#26C6DA"]
    for i, bc in enumerate(book_colors):
        bk = FancyBboxPatch((0.62+i*0.05, 0.57), 0.04, 0.12,
                             boxstyle="round,pad=0.002", color=bc, zorder=4)
        ax.add_patch(bk)
    # lamp
    ax.plot([0.15, 0.15], [0.43, 0.70], color="#757575", lw=3, zorder=3)
    lamp = FancyBboxPatch((0.09, 0.70), 0.12, 0.08, boxstyle="round,pad=0.01",
                           color="#FFF9C4", zorder=4)
    ax.add_patch(lamp)
    ax.add_patch(Circle((0.15, 0.74), 0.025, color="#FDD835", zorder=5))
    # baby sitting, reading big book
    baby_face(ax, 0.38, 0.42, r=0.10)
    body = FancyBboxPatch((0.29, 0.22), 0.18, 0.20, boxstyle="round,pad=0.02",
                           color="#CE93D8", zorder=4)
    ax.add_patch(body)
    # big open book
    book_l = FancyBboxPatch((0.18, 0.14), 0.18, 0.13, boxstyle="round,pad=0.01",
                              color="white", zorder=5, ec="#BDBDBD")
    book_r = FancyBboxPatch((0.36, 0.14), 0.18, 0.13, boxstyle="round,pad=0.01",
                              color="#FFFDE7", zorder=5, ec="#BDBDBD")
    ax.add_patch(book_l)
    ax.add_patch(book_r)
    ax.plot([0.36, 0.36], [0.14, 0.27], color="#9E9E9E", lw=1.5, zorder=6)
    # text lines in book
    for ly in [0.21, 0.19, 0.17]:
        ax.plot([0.20, 0.34], [ly, ly], color="#BDBDBD", lw=1, zorder=6)
        ax.plot([0.38, 0.52], [ly, ly], color="#BDBDBD", lw=1, zorder=6)
    # tiny star on book page
    ax.text(0.45, 0.22, "⭐", fontsize=10, ha='center', zorder=7)
    # arms holding book
    ax.plot([0.29, 0.22], [0.30, 0.22], color="#FDDBB4", lw=5, zorder=5)
    ax.plot([0.47, 0.54], [0.30, 0.22], color="#FDDBB4", lw=5, zorder=5)


def scene_dancing(ax):
    ax.set_facecolor("#F3E5F5")
    draw_sky(ax, "#E8EAF6", 0.35)
    draw_ground(ax, "#CE93D8", 0.25)
    # disco ball / sparkles
    ax.add_patch(Circle((0.5, 0.88), 0.06, color="#E0E0E0", zorder=3))
    for a in range(0, 360, 30):
        r = np.radians(a)
        ax.add_patch(Circle((0.5+0.06*np.cos(r), 0.88+0.06*np.sin(r)),
                             0.01, color="#BDBDBD", zorder=4))
    # sparkle rays
    for a in range(0, 360, 45):
        r = np.radians(a)
        x1, y1 = 0.5+0.07*np.cos(r), 0.88+0.07*np.sin(r)
        x2, y2 = 0.5+0.13*np.cos(r), 0.88+0.13*np.sin(r)
        ax.plot([x1, x2], [y1, y2], color="#FDD835", lw=1.5, zorder=3, alpha=0.8)
    # music notes floating
    for (mx, my, mn) in [(0.12,0.75,"♪"),(0.20,0.82,"♫"),(0.75,0.78,"♪"),
                          (0.85,0.70,"♫"),(0.30,0.90,"♩")]:
        ax.text(mx, my, mn, fontsize=16, color="#9C27B0", ha='center', zorder=4, alpha=0.8)
    # mama dancing
    adult_face(ax, 0.32, 0.68, r=0.12, is_mama=True)
    body_m = FancyBboxPatch((0.23, 0.42), 0.18, 0.26, boxstyle="round,pad=0.02",
                             color="#EC407A", zorder=4)
    ax.add_patch(body_m)
    # skirt flare
    skirt = plt.Polygon([[0.23,0.42],[0.41,0.42],[0.48,0.25],[0.16,0.25]],
                          color="#EC407A", zorder=4)
    ax.add_patch(skirt)
    # mama arms up dancing
    ax.plot([0.23, 0.12], [0.56, 0.66], color="#FDDBB4", lw=6, zorder=5)
    ax.plot([0.41, 0.52], [0.56, 0.66], color="#FDDBB4", lw=6, zorder=5)
    # baby dancing
    baby_face(ax, 0.65, 0.58, r=0.10)
    body_b = FancyBboxPatch((0.57, 0.35), 0.16, 0.23, boxstyle="round,pad=0.02",
                             color="#7E57C2", zorder=4)
    ax.add_patch(body_b)
    ax.plot([0.57, 0.47], [0.48, 0.56], color="#FDDBB4", lw=5, zorder=5)
    ax.plot([0.73, 0.83], [0.48, 0.56], color="#FDDBB4", lw=5, zorder=5)
    # legs dancing
    ax.plot([0.61, 0.54], [0.35, 0.25], color="#FDDBB4", lw=5, zorder=5)
    ax.plot([0.69, 0.78], [0.35, 0.25], color="#FDDBB4", lw=5, zorder=5)


def scene_hug(ax):
    ax.set_facecolor("#FFEBEE")
    draw_sky(ax, "#FFEBEE", 0.40)
    draw_ground(ax, "#FFCDD2", 0.25)
    # big heart background
    ax.text(0.5, 0.60, "❤", fontsize=120, ha='center', va='center',
            color="#FFCDD2", zorder=1, alpha=0.5)
    # floating hearts
    for (hx, hy, hs) in [(0.12,0.80,20),(0.88,0.75,16),(0.20,0.65,12),
                          (0.80,0.85,14),(0.50,0.92,18)]:
        ax.text(hx, hy, "❤", fontsize=hs, ha='center', color="#EF5350",
                zorder=4, alpha=0.7)
    # baba
    adult_face(ax, 0.42, 0.70, r=0.13, is_mama=False)
    body_dad = FancyBboxPatch((0.30, 0.42), 0.24, 0.28, boxstyle="round,pad=0.02",
                               color="#42A5F5", zorder=4)
    ax.add_patch(body_dad)
    # baba's arms wrapping baby
    ax.plot([0.30, 0.20], [0.55, 0.50], color="#FDDBB4", lw=7, zorder=5)
    ax.plot([0.54, 0.64], [0.55, 0.50], color="#FDDBB4", lw=7, zorder=5)
    ax.plot([0.20, 0.62], [0.50, 0.50], color="#FDDBB4", lw=5, zorder=5, alpha=0.8)
    # baby being hugged
    baby_face(ax, 0.60, 0.60, r=0.09)
    body_b = FancyBboxPatch((0.52, 0.42), 0.16, 0.18, boxstyle="round,pad=0.02",
                             color="#A5D6A7", zorder=6)
    ax.add_patch(body_b)
    # baby happy arms up
    ax.plot([0.52, 0.44], [0.52, 0.59], color="#FDDBB4", lw=5, zorder=7)
    ax.plot([0.68, 0.74], [0.52, 0.59], color="#FDDBB4", lw=5, zorder=7)


SCENE_FUNCS = {
    "cover": scene_cover,
    "icecream": scene_icecream,
    "sleeping": scene_sleeping,
    "bath": scene_bath,
    "toys": scene_toys,
    "cooking": scene_cooking,
    "park": scene_park,
    "reading": scene_reading,
    "dancing": scene_dancing,
    "hug": scene_hug,
}


def render_scene_to_image(page):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(page["bg"])
    ax.set_facecolor(page["bg"])
    SCENE_FUNCS[page["scene"]](ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',
                facecolor=page["bg"], dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def make_pdf(output_path):
    c = canvas.Canvas(output_path, pagesize=A4)

    for i, page in enumerate(PAGES):
        bg = HexColor(page["bg"])
        c.setFillColor(bg)
        c.rect(0, 0, W, H, fill=1, stroke=0)

        is_cover = page["scene"] == "cover"

        if is_cover:
            # Decorative border
            for offset, alpha_col in [(8, "#F48FB1"), (16, "#90CAF9"), (24, "#A5D6A7")]:
                c.setStrokeColor(HexColor(alpha_col))
                c.setLineWidth(4)
                c.roundRect(offset, offset, W - 2*offset, H - 2*offset, 15, stroke=1, fill=0)

            # Title banner
            banner_h = 90
            banner_y = H - 150
            c.setFillColor(HexColor("#FF8F00"))
            c.roundRect(30, banner_y, W - 60, banner_h, 20, fill=1, stroke=0)

            c.setFillColor(HexColor("#FFFFFF"))
            c.setFont("Helvetica-Bold", 36)
            c.drawCentredString(W / 2, banner_y + 52, "Il Mio Libro")
            c.setFont("Helvetica-Bold", 26)
            c.drawCentredString(W / 2, banner_y + 18, "di Zoraiz")

            # Image
            img_buf = render_scene_to_image(page)
            img = ImageReader(img_buf)
            img_size = 340
            img_x = (W - img_size) / 2
            img_y = H - 520
            c.roundRect(img_x - 8, img_y - 8, img_size + 16, img_size + 16, 18,
                        fill=0, stroke=1)
            c.setStrokeColor(HexColor("#FF8F00"))
            c.setLineWidth(3)
            c.drawImage(img, img_x, img_y, width=img_size, height=img_size,
                        mask='auto', preserveAspectRatio=True)

            # Italian sentence
            c.setFillColor(HexColor("#6A1B9A"))
            c.setFont("Helvetica-Bold", 22)
            c.drawCentredString(W / 2, img_y - 50, page["title"])
            c.setFont("Helvetica", 18)
            c.setFillColor(HexColor("#E65100"))
            c.drawCentredString(W / 2, img_y - 80, page["subtitle"])

            # Page number dots (decorative)
            for pi in range(len(PAGES)):
                dot_col = "#FF8F00" if pi == i else "#E0E0E0"
                c.setFillColor(HexColor(dot_col))
                c.circle(W/2 - (len(PAGES)-1)*8 + pi*16, 55, 5, fill=1, stroke=0)

        else:
            # Page number bubble top-right
            c.setFillColor(HexColor("#FF8F00"))
            c.circle(W - 42, H - 42, 22, fill=1, stroke=0)
            c.setFillColor(HexColor("#FFFFFF"))
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(W - 42, H - 47, str(i))

            # Decorative top border
            c.setStrokeColor(HexColor("#FF8F00"))
            c.setLineWidth(4)
            c.line(30, H - 30, W - 30, H - 30)

            # Image — large, centred
            img_buf = render_scene_to_image(page)
            img = ImageReader(img_buf)
            img_size = 380
            img_x = (W - img_size) / 2
            img_y = H - 100 - img_size

            # Shadow
            c.setFillColor(HexColor("#E0E0E0"))
            c.roundRect(img_x + 6, img_y - 6, img_size, img_size, 18, fill=1, stroke=0)
            # Frame
            c.setFillColor(HexColor("#FFFFFF"))
            c.setStrokeColor(HexColor("#FF8F00"))
            c.setLineWidth(4)
            c.roundRect(img_x - 10, img_y - 10, img_size + 20, img_size + 20, 20,
                        fill=1, stroke=1)
            c.drawImage(img, img_x, img_y, width=img_size, height=img_size,
                        mask='auto', preserveAspectRatio=True)

            # Italian main sentence — big and bold
            sentence_y = img_y - 60
            c.setFillColor(HexColor("#1A237E"))
            c.setFont("Helvetica-Bold", 26)
            c.drawCentredString(W / 2, sentence_y, page["title"])

            # Subtitle
            c.setFillColor(HexColor("#E65100"))
            c.setFont("Helvetica", 20)
            c.drawCentredString(W / 2, sentence_y - 35, page["subtitle"])

            # Bottom wavy decoration dots
            for di in range(20):
                dot_y = 35 + 8 * np.sin(di * 0.8)
                c.setFillColor(HexColor("#FF8F00" if di % 2 == 0 else "#42A5F5"))
                c.circle(30 + di * 28, dot_y, 5, fill=1, stroke=0)

        c.showPage()

    c.save()
    print(f"PDF saved to {output_path}")


if __name__ == "__main__":
    make_pdf("/home/user/claude/zoraiz_libro.pdf")
