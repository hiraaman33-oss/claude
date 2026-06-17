import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arc
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw
import io

W, H = A4

PAGES = [
    # 1
    dict(line1="Zoraiz e la mama fanno un picnic.", line2="Mangiano gli snack insieme!",
         emoji="🧺", bg="#FFF8E1", acc="#F9A825", icon="picnic"),
    # 2
    dict(line1="Zoraiz guida la macchinina nel parco.", line2="Che divertimento!",
         emoji="🚗", bg="#E3F2FD", acc="#1565C0", icon="car"),
    # 3
    dict(line1="Zoraiz incontra Spider-Man!", line2="Zoraiz ama Spider-Man!",
         emoji="🕷️", bg="#FFEBEE", acc="#C62828", icon="spider"),
    # 4
    dict(line1="Zoraiz mangia il gelato.", line2="Anche d'inverno, che freddo!",
         emoji="🍦", bg="#E8F5E9", acc="#2E7D32", icon="icecream"),
    # 5
    dict(line1="Zoraiz e la mama mostrano i giocattoli.", line2="Zoraiz vuole il cagnolino!",
         emoji="🐶", bg="#FCE4EC", acc="#AD1457", icon="toys"),
    # 6
    dict(line1="Zoraiz gioca allo scivolo con l'amico.", line2="Che bello il giardino!",
         emoji="🛝", bg="#E8F5E9", acc="#388E3C", icon="slide"),
    # 7
    dict(line1="Zoraiz suona il pianoforte.", line2="Zoraiz ama la musica!",
         emoji="🎹", bg="#EDE7F6", acc="#512DA8", icon="piano"),
    # 8
    dict(line1="Zoraiz studia e scrive sul libro.", line2="Bravo Zoraiz!",
         emoji="📚", bg="#E0F7FA", acc="#00838F", icon="study"),
    # 9
    dict(line1="È il compleanno di Zoraiz!", line2="La nonna porta la torta. Auguri!",
         emoji="🎂", bg="#FFF9C4", acc="#F57F17", icon="birthday"),
    # 10
    dict(line1="Zoraiz mangia la pizza.", line2="La pizza è il piatto preferito di Zoraiz!",
         emoji="🍕", bg="#FBE9E7", acc="#BF360C", icon="pizza"),
    # 11
    dict(line1="Zoraiz va a scuola.", line2="Che bello lo zaino blu!",
         emoji="🎒", bg="#E3F2FD", acc="#1565C0", icon="school"),
    # 12
    dict(line1="Zoraiz e la mama sono sulla nave.", line2="Guardano il mare insieme!",
         emoji="🚢", bg="#E0F7FA", acc="#006064", icon="ship"),
    # 13
    dict(line1="Zoraiz ha un anno e dorme nel passeggino.", line2="Che tenero!",
         emoji="😴", bg="#EDE7F6", acc="#4527A0", icon="sleep"),
    # 14
    dict(line1="Benvenuto Zoraiz! Sei appena nato.", line2="Zoraiz vuole il latte della mama!",
         emoji="👶", bg="#FCE4EC", acc="#C62828", icon="newborn"),
    # 15
    dict(line1="Zoraiz piange perché vuole la macchina.", line2="La mama lo consola!",
         emoji="😢", bg="#FFFDE7", acc="#F57F17", icon="cry"),
    # 16
    dict(line1="Zoraiz e baba hanno lo stesso vestito.", line2="Baba vuole tanto bene a Zoraiz!",
         emoji="👔", bg="#ECEFF1", acc="#37474F", icon="matching"),
    # 17
    dict(line1="Zoraiz e baba aspettano il treno.", line2="Zoraiz abbraccia il suo baba!",
         emoji="🚂", bg="#E8F5E9", acc="#1B5E20", icon="train"),
    # 18
    dict(line1="La mama e il baba di Zoraiz sono felici.", line2="Ti vogliamo tanto bene, Zoraiz!",
         emoji="❤️", bg="#FCE4EC", acc="#AD1457", icon="parents"),
    # 19
    dict(line1="Zoraiz compie 2 anni!", line2="Il baba gli fa la sorpresa. Auguri Zoraiz!",
         emoji="🎉", bg="#FFF9C4", acc="#E65100", icon="birthday2"),
    # 20
    dict(line1="Il nonno e la nonna festeggiano il compleanno.", line2="Zoraiz li ama tanto!",
         emoji="🎂", bg="#EFEBE9", acc="#4E342E", icon="grandparents_bday"),
    # 21
    dict(line1="Questi sono i nonni di Zoraiz.", line2="La nonna e il nonno lo amano tanto!",
         emoji="👴", bg="#F1F8E9", acc="#33691E", icon="grandparents"),
    # 22
    dict(line1="Il nonno, la nonna e la zia di Zoraiz.", line2="Sono i genitori del baba. Lo amano tanto!",
         emoji="👨‍👩‍👧", bg="#FFF3E0", acc="#E65100", icon="family2"),
    # 23
    dict(line1="Zoraiz e la mama indossano il vestito pakistano.", line2="Che belli insieme!",
         emoji="👗", bg="#FCE4EC", acc="#880E4F", icon="dress"),
]

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))

# ── icon drawing helpers ──────────────────────────────────────────────────────

def draw_sun(ax, x=0.82,y=0.82,r=0.07,c="#FDD835"):
    ax.add_patch(Circle((x,y),r,color=c,zorder=3))
    for a in range(0,360,45):
        rd=np.radians(a)
        ax.plot([x+r*1.15*np.cos(rd),x+r*1.6*np.cos(rd)],
                [y+r*1.15*np.sin(rd),y+r*1.6*np.sin(rd)],color=c,lw=2,zorder=3)

def draw_stars(ax,n=8,color="#FDD835"):
    np.random.seed(42)
    for _ in range(n):
        x,y=np.random.uniform(0.05,0.95),np.random.uniform(0.65,0.95)
        ax.text(x,y,"★",fontsize=10,color=color,ha='center',va='center',zorder=3,alpha=0.7)

def draw_hearts(ax,positions,color="#E91E63",size=16):
    for x,y in positions:
        ax.text(x,y,"♥",fontsize=size,color=color,ha='center',va='center',zorder=4)

def stick_figure(ax,cx,cy,scale=0.10,color="#5D4037",shirt="#EF5350"):
    # head
    ax.add_patch(Circle((cx,cy+scale*1.4),scale*0.35,color="#FDDBB4",zorder=5))
    # body
    ax.plot([cx,cx],[cy+scale*0.5,cy+scale*1.05],color=shirt,lw=8,zorder=5,solid_capstyle='round')
    # arms
    ax.plot([cx-scale*0.6,cx+scale*0.6],[cy+scale*0.8,cy+scale*0.8],color=shirt,lw=5,zorder=5,solid_capstyle='round')
    # legs
    ax.plot([cx,cx-scale*0.4],[cy+scale*0.5,cy],color=color,lw=6,zorder=5,solid_capstyle='round')
    ax.plot([cx,cx+scale*0.4],[cy+scale*0.5,cy],color=color,lw=6,zorder=5,solid_capstyle='round')

def small_figure(ax,cx,cy,color="#42A5F5"):
    s=0.07
    ax.add_patch(Circle((cx,cy+s*1.4),s*0.30,color="#FDDBB4",zorder=5))
    ax.plot([cx,cx],[cy+s*0.5,cy+s*1.1],color=color,lw=6,zorder=5,solid_capstyle='round')
    ax.plot([cx-s*0.5,cx+s*0.5],[cy+s*0.8,cy+s*0.8],color=color,lw=4,zorder=5,solid_capstyle='round')
    ax.plot([cx,cx-s*0.35],[cy+s*0.5,cy],color="#5D4037",lw=5,zorder=5,solid_capstyle='round')
    ax.plot([cx,cx+s*0.35],[cy+s*0.5,cy],color="#5D4037",lw=5,zorder=5,solid_capstyle='round')

def scene_icon(ax, icon, acc):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

    if icon == "picnic":
        ax.set_facecolor("#FFF8E1")
        draw_sun(ax)
        # grass
        ax.add_patch(FancyBboxPatch((0,0),1,0.28,boxstyle="square",color="#A5D6A7",zorder=1))
        # blanket
        ax.add_patch(FancyBboxPatch((0.15,0.22),0.70,0.20,boxstyle="round,pad=0.02",color="white",zorder=2,ec="#BDBDBD",lw=2))
        # basket
        ax.add_patch(FancyBboxPatch((0.18,0.26),0.12,0.10,boxstyle="round,pad=0.01",color="#8D6E63",zorder=3))
        ax.plot([0.18,0.30],[0.36,0.36],color="#6D4C41",lw=3,zorder=4)
        ax.plot([0.24,0.24],[0.36,0.40],color="#6D4C41",lw=3,zorder=4)
        # food
        ax.add_patch(Circle((0.50,0.30),0.04,color="#FFA726",zorder=3))
        ax.add_patch(Circle((0.58,0.29),0.03,color="#EF5350",zorder=3))
        # mama figure
        stick_figure(ax,0.30,0.33,scale=0.09,shirt="#F48FB1")
        # baby figure
        small_figure(ax,0.62,0.33,color="#80DEEA")
        # trees
        for tx in [0.05,0.90]:
            ax.add_patch(FancyBboxPatch((tx-0.02,0.10),0.04,0.15,boxstyle="square",color="#795548",zorder=2))
            ax.add_patch(Circle((tx,0.28),0.08,color="#4CAF50",zorder=3))

    elif icon == "car":
        ax.set_facecolor("#E3F2FD")
        draw_sun(ax)
        ax.add_patch(FancyBboxPatch((0,0),1,0.25,boxstyle="square",color="#C8E6C9",zorder=1))
        # electric ride-on car (green frog style)
        car = FancyBboxPatch((0.20,0.22),0.60,0.28,boxstyle="round,pad=0.04",color="#81C784",zorder=3)
        ax.add_patch(car)
        ax.add_patch(FancyBboxPatch((0.28,0.38),0.44,0.18,boxstyle="round,pad=0.03",color="#A5D6A7",zorder=4))
        # wheels
        for wx in [0.28,0.72]:
            ax.add_patch(Circle((wx,0.22),0.07,color="#37474F",zorder=5))
            ax.add_patch(Circle((wx,0.22),0.03,color="#90A4AE",zorder=6))
        # eyes on car
        ax.add_patch(Circle((0.38,0.48),0.05,color="#29B6F6",zorder=5))
        ax.add_patch(Circle((0.62,0.48),0.05,color="#29B6F6",zorder=5))
        # baby in car
        ax.add_patch(Circle((0.50,0.55),0.09,color="#FDDBB4",zorder=6))
        ax.add_patch(Circle((0.47,0.57),0.015,color="#333",zorder=7))
        ax.add_patch(Circle((0.53,0.57),0.015,color="#333",zorder=7))
        ax.add_patch(Arc((0.50,0.53),0.06,0.04,angle=0,theta1=200,theta2=340,color="#c0392b",lw=2,zorder=7))

    elif icon == "spider":
        ax.set_facecolor("#FFEBEE")
        # web
        for r in [0.15,0.28,0.42]:
            ax.add_patch(Circle((0.5,0.75),r,fill=False,ec="#EF9A9A",lw=1.5,zorder=2))
        for a in range(0,360,45):
            rd=np.radians(a)
            ax.plot([0.5,0.5+0.45*np.cos(rd)],[0.75,0.75+0.45*np.sin(rd)],color="#EF9A9A",lw=1,zorder=2)
        # spiderman
        ax.add_patch(Circle((0.70,0.70),0.13,color="#E53935",zorder=5))
        # web pattern on suit
        ax.add_patch(FancyBboxPatch((0.58,0.42),0.24,0.26,boxstyle="round,pad=0.02",color="#E53935",zorder=4))
        # blue stripe
        ax.add_patch(FancyBboxPatch((0.60,0.44),0.20,0.22,boxstyle="round,pad=0.01",color="#1565C0",zorder=4,alpha=0.5))
        # zoraiz
        small_figure(ax,0.32,0.32,color="#3F51B5")
        ax.text(0.50,0.25,"Ciao, Spider-Man!",fontsize=8,ha='center',color="#C62828",fontweight='bold',zorder=6)

    elif icon == "icecream":
        ax.set_facecolor("#E8F5E9")
        # snowflakes
        for (sx,sy) in [(0.15,0.80),(0.30,0.90),(0.70,0.85),(0.85,0.75),(0.50,0.95)]:
            ax.text(sx,sy,"❄",fontsize=14,color="#90CAF9",ha='center',va='center',zorder=3)
        # big cone
        tri=plt.Polygon([[0.5,0.15],[0.38,0.48],[0.62,0.48]],color="#D7CCC8",zorder=4)
        ax.add_patch(tri)
        ax.add_patch(Circle((0.50,0.52),0.10,color="#F8BBD9",zorder=5))
        ax.add_patch(Circle((0.50,0.59),0.08,color="#F3E5F5",zorder=6))
        ax.add_patch(Circle((0.50,0.65),0.06,color="#FFF9C4",zorder=7))
        # baby holding cone
        small_figure(ax,0.50,0.20,color="#EF5350")

    elif icon == "toys":
        ax.set_facecolor("#FCE4EC")
        draw_sun(ax,0.82,0.82,c="#FDD835")
        # toy dog
        ax.add_patch(Circle((0.65,0.55),0.10,color="white",zorder=4,ec="#90A4AE",lw=2))
        ax.add_patch(Circle((0.58,0.61),0.05,color="#90A4AE",zorder=4))
        ax.add_patch(Circle((0.72,0.61),0.05,color="#90A4AE",zorder=4))
        ax.add_patch(Circle((0.62,0.56),0.02,color="#212121",zorder=5))
        ax.add_patch(Circle((0.68,0.56),0.02,color="#212121",zorder=5))
        # pink pig
        ax.add_patch(Circle((0.30,0.55),0.08,color="#F48FB1",zorder=4))
        ax.add_patch(Circle((0.30,0.60),0.05,color="#F48FB1",zorder=5))
        ax.add_patch(Circle((0.29,0.62),0.015,color="#C2185B",zorder=6))
        ax.add_patch(Circle((0.32,0.62),0.015,color="#C2185B",zorder=6))
        # baby reaching
        small_figure(ax,0.48,0.32,color="#7C4DFF")
        ax.plot([0.48,0.60],[0.42,0.52],color="#FDDBB4",lw=4,zorder=6)

    elif icon == "slide":
        ax.set_facecolor("#F1F8E9")
        draw_sun(ax)
        ax.add_patch(FancyBboxPatch((0,0),1,0.22,boxstyle="square",color="#A5D6A7",zorder=1))
        # slide structure
        ax.add_patch(FancyBboxPatch((0.15,0.22),0.08,0.45,boxstyle="round,pad=0.01",color="#90A4AE",zorder=3))
        slide=plt.Polygon([[0.23,0.67],[0.70,0.22],[0.78,0.22],[0.31,0.67]],color="#EF5350",zorder=4)
        ax.add_patch(slide)
        # ladder
        ax.add_patch(FancyBboxPatch((0.15,0.22),0.08,0.45,boxstyle="square",color="#90A4AE",zorder=3,alpha=0.3))
        for ry in [0.32,0.42,0.52,0.62]:
            ax.plot([0.15,0.23],[ry,ry],color="#607D8B",lw=3,zorder=4)
        # kids
        small_figure(ax,0.21,0.60,color="#7E57C2")
        small_figure(ax,0.72,0.28,color="#FF7043")

    elif icon == "piano":
        ax.set_facecolor("#EDE7F6")
        draw_stars(ax,6,"#CE93D8")
        # piano body
        ax.add_patch(FancyBboxPatch((0.08,0.32),0.84,0.48,boxstyle="round,pad=0.02",color="#212121",zorder=3))
        # keys
        for i in range(7):
            ax.add_patch(FancyBboxPatch((0.11+i*0.11,0.34),0.09,0.24,boxstyle="round,pad=0.005",color="white",zorder=4,ec="#9E9E9E",lw=0.5))
        for i,pos in enumerate([0,1,3,4,5]):
            ax.add_patch(FancyBboxPatch((0.17+pos*0.11,0.40),0.06,0.16,boxstyle="round,pad=0.002",color="#212121",zorder=5))
        # music notes
        for (nx,ny) in [(0.20,0.90),(0.50,0.93),(0.78,0.88)]:
            ax.text(nx,ny,"♪",fontsize=16,color="#9C27B0",ha='center',zorder=6)
        # baby sitting
        small_figure(ax,0.50,0.18,color="#F44336")

    elif icon == "study":
        ax.set_facecolor("#E0F7FA")
        # bed
        ax.add_patch(FancyBboxPatch((0.10,0.22),0.80,0.40,boxstyle="round,pad=0.02",color="#8D6E63",zorder=2))
        ax.add_patch(FancyBboxPatch((0.13,0.38),0.74,0.22,boxstyle="round,pad=0.02",color="#CE93D8",zorder=3))
        # book
        ax.add_patch(FancyBboxPatch((0.30,0.42),0.20,0.14,boxstyle="round,pad=0.01",color="white",zorder=5,ec="#BDBDBD"))
        ax.add_patch(FancyBboxPatch((0.50,0.42),0.20,0.14,boxstyle="round,pad=0.01",color="#FFF9C4",zorder=5,ec="#BDBDBD"))
        ax.plot([0.50,0.50],[0.42,0.56],color="#9E9E9E",lw=1.5,zorder=6)
        for ly in [0.51,0.49,0.47]:
            ax.plot([0.32,0.48],[ly,ly],color="#E0E0E0",lw=1,zorder=6)
        # pen
        ax.plot([0.43,0.50],[0.50,0.42],color="#2196F3",lw=3,zorder=7)
        # baby lying
        ax.add_patch(Circle((0.50,0.62),0.09,color="#FDDBB4",zorder=6))
        ax.add_patch(Circle((0.47,0.63),0.015,color="#333",zorder=7))
        ax.add_patch(Circle((0.53,0.63),0.015,color="#333",zorder=7))
        ax.add_patch(Arc((0.50,0.60),0.06,0.04,angle=0,theta1=200,theta2=340,color="#c0392b",lw=2,zorder=7))
        # balloons
        for (bx,bc) in [(0.78,"#EF5350"),(0.87,"#42A5F5"),(0.93,"#66BB6A")]:
            ax.add_patch(Circle((bx,0.82),0.05,color=bc,zorder=4))
            ax.plot([bx,bx-0.02],[0.77,0.65],color=bc,lw=1,zorder=4)

    elif icon == "birthday":
        ax.set_facecolor("#FFF9C4")
        draw_hearts(ax,[(0.15,0.85),(0.85,0.85),(0.50,0.92)],color="#E91E63",size=18)
        # table
        ax.add_patch(FancyBboxPatch((0.15,0.22),0.70,0.08,boxstyle="round,pad=0.01",color="#BCAAA4",zorder=3))
        ax.plot([0.25,0.25],[0.14,0.22],color="#A1887F",lw=6,zorder=3)
        ax.plot([0.75,0.75],[0.14,0.22],color="#A1887F",lw=6,zorder=3)
        # cake
        ax.add_patch(FancyBboxPatch((0.32,0.30),0.36,0.22,boxstyle="round,pad=0.02",color="white",zorder=4,ec="#BDBDBD"))
        # frosting
        ax.add_patch(FancyBboxPatch((0.32,0.46),0.36,0.08,boxstyle="round,pad=0.01",color="#F8BBD9",zorder=5))
        # fruits on top
        for (fx,fc) in [(0.38,"#EF5350"),(0.44,"#FFA726"),(0.50,"#4CAF50"),(0.56,"#EF5350"),(0.62,"#9C27B0")]:
            ax.add_patch(Circle((fx,0.54),0.025,color=fc,zorder=6))
        # candles
        for cx in [0.42,0.50,0.58]:
            ax.plot([cx,cx],[0.52,0.62],color="#FFF176",lw=4,zorder=6)
            ax.add_patch(Circle((cx,0.63),0.025,color="#FF8F00",zorder=7))
        # nonna
        stick_figure(ax,0.72,0.32,scale=0.09,shirt="#E53935")
        # zoraiz
        small_figure(ax,0.28,0.32,color="#FF9800")
        ax.text(0.50,0.82,"Auguri!",fontsize=14,ha='center',color="#AD1457",fontweight='bold',zorder=8)

    elif icon == "pizza":
        ax.set_facecolor("#FBE9E7")
        # pizza
        tri=plt.Polygon([[0.5,0.72],[0.22,0.25],[0.78,0.25]],color="#FFA726",zorder=3)
        ax.add_patch(tri)
        ax.add_patch(Arc((0.5,0.25),0.56,0.56,angle=0,theta1=0,theta2=180,color="#EF5350",lw=8,zorder=4))
        # cheese
        for (px,py) in [(0.42,0.42),(0.55,0.38),(0.48,0.55),(0.38,0.35),(0.60,0.50)]:
            ax.add_patch(Circle((px,py),0.035,color="#FDD835",zorder=5))
        # toppings
        for (px,py) in [(0.45,0.48),(0.55,0.44),(0.50,0.36)]:
            ax.add_patch(Circle((px,py),0.02,color="#B71C1C",zorder=6))
        # zoraiz
        small_figure(ax,0.50,0.15,color="#3F51B5")
        ax.text(0.50,0.88,"Che buona!",fontsize=12,ha='center',color="#BF360C",fontweight='bold',zorder=7)

    elif icon == "school":
        ax.set_facecolor("#E3F2FD")
        draw_sun(ax)
        # school building
        ax.add_patch(FancyBboxPatch((0.20,0.30),0.60,0.45,boxstyle="square",color="#FFCCBC",zorder=2))
        ax.add_patch(plt.Polygon([[0.20,0.75],[0.50,0.92],[0.80,0.75]],color="#EF5350",zorder=3))
        # windows
        for wx in [0.30,0.60]:
            ax.add_patch(FancyBboxPatch((wx,0.50),0.12,0.12,boxstyle="round,pad=0.01",color="#B3E5FC",zorder=4,ec="#90CAF9"))
        # door
        ax.add_patch(FancyBboxPatch((0.43,0.30),0.14,0.20,boxstyle="round,pad=0.01",color="#8D6E63",zorder=4))
        # backpack kid
        small_figure(ax,0.50,0.18,color="#1565C0")
        # backpack
        ax.add_patch(FancyBboxPatch((0.52,0.22),0.07,0.10,boxstyle="round,pad=0.01",color="#1565C0",zorder=6))
        ax.text(0.50,0.08,"Ciao scuola!",fontsize=11,ha='center',color="#1565C0",fontweight='bold',zorder=7)

    elif icon == "ship":
        ax.set_facecolor("#E0F7FA")
        # sea
        ax.add_patch(FancyBboxPatch((0,0),1,0.45,boxstyle="square",color="#0288D1",zorder=1))
        # waves
        for wy in [0.35,0.40,0.45]:
            for wx in np.arange(0,1,0.2):
                ax.add_patch(Arc((wx+0.1,wy),0.2,0.06,angle=0,theta1=0,theta2=180,color="white",lw=2,zorder=2))
        # ship
        ship=plt.Polygon([[0.15,0.42],[0.85,0.42],[0.80,0.28],[0.20,0.28]],color="white",zorder=3)
        ax.add_patch(ship)
        ax.add_patch(FancyBboxPatch((0.35,0.28),0.30,0.22,boxstyle="round,pad=0.01",color="#BBDEFB",zorder=4))
        # chimney
        ax.plot([0.50,0.50],[0.50,0.68],color="#616161",lw=8,zorder=5)
        ax.add_patch(Circle((0.50,0.68),0.04,color="#F44336",zorder=6))
        # railing
        ax.plot([0.18,0.82],[0.44,0.44],color="#EF5350",lw=3,zorder=5)
        # mama & baby
        stick_figure(ax,0.58,0.47,scale=0.08,shirt="#1565C0")
        small_figure(ax,0.44,0.47,color="#80DEEA")
        ax.text(0.50,0.88,"Che bel mare!",fontsize=11,ha='center',color="#006064",fontweight='bold',zorder=7)

    elif icon == "sleep":
        ax.set_facecolor("#EDE7F6")
        # night
        ax.add_patch(FancyBboxPatch((0,0.45),1,0.55,boxstyle="square",color="#1A237E",zorder=1))
        draw_stars(ax,10,"#FFF9C4")
        ax.add_patch(Circle((0.80,0.82),0.07,color="#FFF9C4",zorder=3))
        ax.add_patch(Circle((0.86,0.84),0.06,color="#1A237E",zorder=4))
        # pram
        ax.add_patch(FancyBboxPatch((0.15,0.22),0.55,0.28,boxstyle="round,pad=0.03",color="#E53935",zorder=3))
        # hood
        ax.add_patch(patches.Wedge((0.70,0.36),0.25,90,270,color="#E53935",zorder=4))
        # wheels
        ax.add_patch(Circle((0.25,0.20),0.05,color="#37474F",zorder=5))
        ax.add_patch(Circle((0.55,0.20),0.05,color="#37474F",zorder=5))
        ax.plot([0.70,0.82],[0.28,0.20],color="#37474F",lw=4,zorder=4)
        ax.add_patch(Circle((0.82,0.20),0.04,color="#37474F",zorder=5))
        # baby sleeping inside
        ax.add_patch(Circle((0.42,0.38),0.08,color="#FDDBB4",zorder=5))
        ax.add_patch(FancyBboxPatch((0.20,0.25),0.10,0.06,boxstyle="round,pad=0.01",color="#81D4FA",zorder=5))
        ax.text(0.55,0.50,"z z z",fontsize=12,color="#CE93D8",fontstyle='italic',ha='center',zorder=6)

    elif icon == "newborn":
        ax.set_facecolor("#FCE4EC")
        draw_hearts(ax,[(0.20,0.88),(0.80,0.88),(0.50,0.93)],color="#E91E63")
        # blanket/crib
        ax.add_patch(FancyBboxPatch((0.10,0.12),0.80,0.52,boxstyle="round,pad=0.04",color="#F3E5F5",zorder=2,ec="#CE93D8",lw=3))
        # baby
        ax.add_patch(Circle((0.50,0.48),0.14,color="#FDDBB4",zorder=4))
        ax.add_patch(patches.Wedge((0.50,0.58),0.10,180,360,color="#90CAF9",zorder=5))
        ax.add_patch(Circle((0.46,0.50),0.02,color="#333",zorder=6))
        ax.add_patch(Circle((0.54,0.50),0.02,color="#333",zorder=6))
        ax.add_patch(Arc((0.50,0.46),0.06,0.04,angle=0,theta1=200,theta2=340,color="#c0392b",lw=2,zorder=6))
        # tiny fists up
        ax.add_patch(Circle((0.36,0.55),0.04,color="#FDDBB4",zorder=5))
        ax.add_patch(Circle((0.64,0.55),0.04,color="#FDDBB4",zorder=5))
        ax.text(0.50,0.78,"Benvenuto!",fontsize=14,ha='center',color="#880E4F",fontweight='bold',zorder=7)
        ax.add_patch(FancyBboxPatch((0.12,0.14),0.76,0.10,boxstyle="round,pad=0.01",color="#EF9A9A",zorder=3))

    elif icon == "cry":
        ax.set_facecolor("#FFFDE7")
        # toy car he wants
        ax.add_patch(FancyBboxPatch((0.58,0.45),0.32,0.18,boxstyle="round,pad=0.02",color="#F44336",zorder=4))
        ax.add_patch(FancyBboxPatch((0.62,0.54),0.24,0.12,boxstyle="round,pad=0.01",color="#EF9A9A",zorder=5))
        for wx in [0.64,0.84]:
            ax.add_patch(Circle((wx,0.43),0.05,color="#37474F",zorder=5))
        ax.text(0.74,0.68,"🚗",fontsize=20,ha='center',zorder=6)
        # crying baby
        small_figure(ax,0.28,0.32,color="#3F51B5")
        # tears
        for (tx,ty) in [(0.25,0.54),(0.30,0.52)]:
            ax.add_patch(Circle((tx,ty),0.018,color="#29B6F6",zorder=6,alpha=0.8))
        ax.text(0.28,0.62,"😢",fontsize=20,ha='center',zorder=7)
        # mama comforting
        stick_figure(ax,0.48,0.32,scale=0.09,shirt="#EC407A")
        ax.plot([0.42,0.34],[0.37,0.40],color="#FDDBB4",lw=5,zorder=6)

    elif icon == "matching":
        ax.set_facecolor("#ECEFF1")
        draw_sun(ax)
        # baba - tall
        ax.add_patch(Circle((0.65,0.72),0.10,color="#FDDBB4",zorder=5))
        ax.plot([0.65,0.65],[0.50,0.62],color="#263238",lw=12,zorder=5,solid_capstyle='round')
        ax.plot([0.52,0.78],[0.58,0.58],color="#263238",lw=8,zorder=5,solid_capstyle='round')
        ax.plot([0.65,0.55],[0.50,0.34],color="#78909C",lw=8,zorder=5,solid_capstyle='round')
        ax.plot([0.65,0.75],[0.50,0.34],color="#78909C",lw=8,zorder=5,solid_capstyle='round')
        # sunglasses on baba
        ax.plot([0.58,0.72],[0.73,0.73],color="#212121",lw=3,zorder=7)
        ax.add_patch(FancyBboxPatch((0.57,0.71),0.06,0.04,boxstyle="round,pad=0.005",color="#212121",zorder=8))
        ax.add_patch(FancyBboxPatch((0.65,0.71),0.06,0.04,boxstyle="round,pad=0.005",color="#212121",zorder=8))
        # baby - small same outfit
        ax.add_patch(Circle((0.35,0.55),0.08,color="#FDDBB4",zorder=5))
        ax.plot([0.35,0.35],[0.36,0.47],color="#263238",lw=9,zorder=5,solid_capstyle='round')
        ax.plot([0.25,0.45],[0.44,0.44],color="#263238",lw=6,zorder=5,solid_capstyle='round')
        ax.plot([0.35,0.27],[0.36,0.24],color="#78909C",lw=6,zorder=5,solid_capstyle='round')
        ax.plot([0.35,0.43],[0.36,0.24],color="#78909C",lw=6,zorder=5,solid_capstyle='round')
        ax.add_patch(FancyBboxPatch((0.29,0.53),0.05,0.03,boxstyle="round,pad=0.003",color="#212121",zorder=8))
        ax.add_patch(FancyBboxPatch((0.36,0.53),0.05,0.03,boxstyle="round,pad=0.003",color="#212121",zorder=8))
        # hand holding
        ax.plot([0.43,0.55],[0.42,0.55],color="#FDDBB4",lw=5,zorder=6)
        draw_hearts(ax,[(0.50,0.82)],color="#E91E63",size=20)

    elif icon == "train":
        ax.set_facecolor("#E8F5E9")
        draw_sun(ax)
        # platform
        ax.add_patch(FancyBboxPatch((0,0.10),1,0.18,boxstyle="square",color="#BDBDBD",zorder=2))
        # train (blue)
        ax.add_patch(FancyBboxPatch((0.05,0.28),0.90,0.38,boxstyle="round,pad=0.02",color="#1565C0",zorder=3))
        # train windows
        for wx in [0.15,0.35,0.55,0.75]:
            ax.add_patch(FancyBboxPatch((wx,0.40),0.16,0.18,boxstyle="round,pad=0.01",color="#B3E5FC",zorder=4,ec="#90CAF9"))
        # front
        ax.add_patch(FancyBboxPatch((0.80,0.28),0.18,0.38,boxstyle="round,pad=0.03",color="#0D47A1",zorder=4))
        ax.add_patch(Circle((0.92,0.42),0.08,color="#B3E5FC",zorder=5))
        # wheels
        for wx in [0.15,0.40,0.65,0.88]:
            ax.add_patch(Circle((wx,0.26),0.05,color="#37474F",zorder=5))
        # baba and zoraiz
        stick_figure(ax,0.42,0.14,scale=0.08,shirt="#1565C0")
        small_figure(ax,0.28,0.14,color="#81D4FA")
        ax.text(0.35,0.05,"In viaggio!",fontsize=11,ha='center',color="#1B5E20",fontweight='bold',zorder=7)
        draw_hearts(ax,[(0.58,0.14)],color="#E91E63",size=14)

    elif icon == "parents":
        ax.set_facecolor("#FCE4EC")
        draw_hearts(ax,[(0.15,0.88),(0.50,0.92),(0.85,0.88),(0.30,0.80),(0.70,0.80)],color="#E91E63",size=18)
        # mama
        ax.add_patch(Circle((0.33,0.68),0.11,color="#FDDBB4",zorder=5))
        ax.plot([0.33,0.33],[0.46,0.57],color="#EC407A",lw=12,zorder=5,solid_capstyle='round')
        ax.plot([0.20,0.46],[0.54,0.54],color="#EC407A",lw=8,zorder=5,solid_capstyle='round')
        # mama hair
        ax.add_patch(patches.Wedge((0.33,0.74),0.12,180,360,color="#3E2723",zorder=6))
        for dx in [-1,1]:
            ax.add_patch(FancyBboxPatch((0.33+dx*0.10,0.55),0.05,0.18,boxstyle="round,pad=0.01",color="#3E2723",zorder=4))
        # baba
        ax.add_patch(Circle((0.67,0.68),0.11,color="#FDDBB4",zorder=5))
        ax.plot([0.67,0.67],[0.46,0.57],color="#37474F",lw=12,zorder=5,solid_capstyle='round')
        ax.plot([0.54,0.80],[0.54,0.54],color="#37474F",lw=8,zorder=5,solid_capstyle='round')
        ax.add_patch(patches.Wedge((0.67,0.74),0.12,180,360,color="#212121",zorder=6))
        # zoraiz between them
        small_figure(ax,0.50,0.40,color="#FFA726")
        ax.text(0.50,0.26,"Ti vogliamo bene!",fontsize=12,ha='center',color="#880E4F",fontweight='bold',zorder=8)

    elif icon == "birthday2":
        ax.set_facecolor("#FFF9C4")
        # big 2
        ax.text(0.50,0.62,"2",fontsize=100,ha='center',va='center',
                color="#E65100",fontweight='bold',zorder=2,alpha=0.15)
        # balloons
        colors=["#E53935","#FB8C00","#FDD835","#43A047","#1E88E5","#8E24AA","#E53935"]
        for i,(bx,bc) in enumerate(zip(np.linspace(0.10,0.90,7),colors)):
            ax.add_patch(Circle((bx,0.80+0.05*np.sin(i)),0.06,color=bc,zorder=4))
            ax.plot([bx,bx-0.01],[0.74,0.55],color=bc,lw=1.5,zorder=4)
        # cake with candle 2
        ax.add_patch(FancyBboxPatch((0.30,0.28),0.40,0.22,boxstyle="round,pad=0.02",color="white",zorder=4,ec="#BDBDBD"))
        ax.add_patch(FancyBboxPatch((0.30,0.44),0.40,0.08,boxstyle="round,pad=0.01",color="#F8BBD9",zorder=5))
        ax.text(0.50,0.52,"2",fontsize=16,ha='center',va='center',color="#E65100",fontweight='bold',zorder=7)
        ax.plot([0.50,0.50],[0.50,0.60],color="#FFF176",lw=5,zorder=6)
        ax.add_patch(Circle((0.50,0.61),0.03,color="#FF8F00",zorder=7))
        # baba blowing horn
        stick_figure(ax,0.28,0.18,scale=0.09,shirt="#1565C0")
        ax.plot([0.28,0.36],[0.30,0.28],color="#FF8F00",lw=4,zorder=6)
        # zoraiz
        small_figure(ax,0.72,0.18,color="#F44336")
        ax.text(0.50,0.10,"Auguri Zoraiz!",fontsize=13,ha='center',color="#E65100",fontweight='bold',zorder=8)

    elif icon == "grandparents_bday":
        ax.set_facecolor("#EFEBE9")
        draw_stars(ax,6,"#FFCC02")
        # table + cake
        ax.add_patch(FancyBboxPatch((0.15,0.22),0.70,0.08,boxstyle="round,pad=0.01",color="#8D6E63",zorder=3))
        ax.add_patch(FancyBboxPatch((0.33,0.30),0.34,0.20,boxstyle="round,pad=0.02",color="#5D4037",zorder=4))
        # candles
        for cx in [0.40,0.50,0.60]:
            ax.plot([cx,cx],[0.50,0.62],color="#FFF176",lw=4,zorder=6)
            ax.add_patch(Circle((cx,0.63),0.025,color="#FF6F00",zorder=7))
        # nonno
        stick_figure(ax,0.30,0.30,scale=0.09,shirt="#ECEFF1")
        ax.add_patch(Circle((0.30,0.50),0.04,color="#9E9E9E",zorder=7))  # white beard hint
        # nonna
        stick_figure(ax,0.70,0.30,scale=0.09,shirt="#9C27B0")
        draw_hearts(ax,[(0.50,0.82)],color="#C62828",size=22)
        ax.text(0.50,0.10,"Auguri nonni!",fontsize=12,ha='center',color="#4E342E",fontweight='bold',zorder=8)

    elif icon == "grandparents":
        ax.set_facecolor("#F1F8E9")
        draw_sun(ax,0.15,0.85,c="#FDD835")
        # trees
        for tx in [0.05,0.88]:
            ax.add_patch(FancyBboxPatch((tx,0.18),0.05,0.25,boxstyle="square",color="#795548",zorder=2))
            ax.add_patch(Circle((tx+0.025,0.46),0.10,color="#388E3C",zorder=3))
        # ami (nonna)
        stick_figure(ax,0.32,0.32,scale=0.09,shirt="#E91E63")
        # mama middle
        stick_figure(ax,0.50,0.36,scale=0.10,shirt="#9C27B0")
        # abu (nonno)
        stick_figure(ax,0.68,0.32,scale=0.09,shirt="#F5F5F5")
        # ground
        ax.add_patch(FancyBboxPatch((0,0),1,0.20,boxstyle="square",color="#8BC34A",zorder=1))
        draw_hearts(ax,[(0.50,0.82)],color="#E91E63",size=18)

    elif icon == "family2":
        ax.set_facecolor("#FFF3E0")
        # nonno
        stick_figure(ax,0.22,0.36,scale=0.09,shirt="#F5F5F5")
        # nonna
        stick_figure(ax,0.42,0.36,scale=0.09,shirt="#80CBC4")
        # zia (phoppo)
        stick_figure(ax,0.65,0.40,scale=0.10,shirt="#CE93D8")
        # zoraiz in front
        small_figure(ax,0.50,0.20,color="#FF9800")
        # christmas tree hint
        tree=plt.Polygon([[0.82,0.22],[0.74,0.45],[0.90,0.45]],color="#388E3C",zorder=3)
        ax.add_patch(tree)
        tree2=plt.Polygon([[0.82,0.36],[0.73,0.58],[0.91,0.58]],color="#2E7D32",zorder=3)
        ax.add_patch(tree2)
        ax.plot([0.82,0.82],[0.22,0.14],color="#795548",lw=5,zorder=4)
        for (ox,oy,oc) in [(0.78,0.50,"#E53935"),(0.84,0.43,"#FDD835"),(0.80,0.56,"#42A5F5")]:
            ax.add_patch(Circle((ox,oy),0.025,color=oc,zorder=5))
        draw_hearts(ax,[(0.30,0.82),(0.70,0.82)],color="#E91E63",size=16)

    elif icon == "dress":
        ax.set_facecolor("#FCE4EC")
        draw_hearts(ax,[(0.20,0.88),(0.50,0.93),(0.80,0.88)],color="#880E4F",size=18)
        # mama in Pakistani dress
        ax.add_patch(Circle((0.38,0.74),0.10,color="#FDDBB4",zorder=5))
        dress=plt.Polygon([[0.28,0.44],[0.48,0.44],[0.55,0.18],[0.21,0.18]],color="#212121",zorder=4)
        ax.add_patch(dress)
        ax.plot([0.38,0.38],[0.44,0.64],color="#212121",lw=10,zorder=5,solid_capstyle='round')
        ax.plot([0.24,0.52],[0.55,0.55],color="#212121",lw=7,zorder=5,solid_capstyle='round')
        # flowers on dress
        for (fx,fy,fc) in [(0.33,0.30,"#E91E63"),(0.42,0.25,"#FF5722"),(0.36,0.20,"#4CAF50")]:
            ax.add_patch(Circle((fx,fy),0.025,color=fc,zorder=5))
        # mama hair
        ax.add_patch(patches.Wedge((0.38,0.80),0.12,180,360,color="#1A237E",zorder=6))
        for dx in [-1,1]:
            ax.add_patch(FancyBboxPatch((0.38+dx*0.09,0.60),0.04,0.20,boxstyle="round,pad=0.01",color="#1A237E",zorder=4))
        # baby in matching colours
        small_figure(ax,0.65,0.32,color="#212121")
        ax.add_patch(FancyBboxPatch((0.66,0.33),0.07,0.12,boxstyle="round,pad=0.01",color="#E91E63",zorder=6,alpha=0.6))
        ax.text(0.50,0.08,"Che belli!",fontsize=13,ha='center',color="#880E4F",fontweight='bold',zorder=8)

    else:
        # fallback
        ax.set_facecolor("#F5F5F5")
        ax.text(0.5,0.5,"♥",fontsize=80,ha='center',va='center',color="#EF5350",zorder=3,alpha=0.3)

# ── render scene to PNG buffer ────────────────────────────────────────────────

def render_scene(page):
    fig, ax = plt.subplots(figsize=(5,5), dpi=110)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    fig.patch.set_facecolor(page["bg"])
    ax.set_facecolor(page["bg"])
    scene_icon(ax, page["icon"], page["acc"])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=page["bg"], dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf

def img_to_reader(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return ImageReader(buf)

# ── PDF pages ─────────────────────────────────────────────────────────────────

def draw_cover(c):
    c.setFillColorRGB(*hex_rgb("#FF8F00"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    # top white arch
    c.setFillColor(white)
    c.roundRect(20, H-160, W-40, 145, 30, fill=1, stroke=0)
    c.setFillColorRGB(*hex_rgb("#FF8F00"))
    c.setFont("Helvetica-Bold", 44)
    c.drawCentredString(W/2, H-80, "Il Mio Libro")
    c.setFont("Helvetica-Bold", 32)
    c.drawCentredString(W/2, H-120, "di Zoraiz ❤")
    # centre heart
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 120)
    c.drawCentredString(W/2, H/2-20, "♥")
    c.setFillColorRGB(*hex_rgb("#FF8F00"))
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(W/2, H/2-20, "ZORAIZ")
    # bottom banner
    c.setFillColor(white)
    c.roundRect(20, 40, W-40, 80, 18, fill=1, stroke=0)
    c.setFillColorRGB(*hex_rgb("#FF8F00"))
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(W/2, 95, "Impariamo l'italiano insieme!")
    c.setFont("Helvetica", 16)
    c.drawCentredString(W/2, 65, f"{len(PAGES)} pagine di avventure")

def draw_page(c, page, num, total):
    bg = hex_rgb(page["bg"])
    acc = hex_rgb(page["acc"])
    c.setFillColorRGB(*bg)
    c.rect(0,0,W,H,fill=1,stroke=0)

    # top bar
    c.setFillColorRGB(*acc)
    c.rect(0, H-56, W, 56, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(W/2, H-36, "Il Mio Libro di Zoraiz")
    # page number
    c.setFillColorRGB(*bg)
    c.circle(W-36, H-28, 18, fill=1, stroke=0)
    c.setFillColorRGB(*acc)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(W-36, H-33, str(num))

    # illustration
    scene_buf = render_scene(page)
    img_size = 390
    ix = (W-img_size)/2
    iy = H - 60 - img_size - 8
    # shadow
    c.setFillColorRGB(0.75,0.75,0.75)
    c.roundRect(ix+6, iy-6, img_size, img_size, 18, fill=1, stroke=0)
    # white frame
    c.setFillColor(white)
    c.setStrokeColorRGB(*acc)
    c.setLineWidth(5)
    c.roundRect(ix-10, iy-10, img_size+20, img_size+20, 22, fill=1, stroke=1)
    c.drawImage(ImageReader(scene_buf), ix, iy, width=img_size, height=img_size, mask='auto')

    # sentence box
    bx=24; by=iy-118; bw=W-48; bh=105
    c.setFillColorRGB(*acc)
    c.roundRect(bx, by, bw, bh, 16, fill=1, stroke=0)
    c.setFillColor(white)
    line1=page["line1"]; line2=page.get("line2","")
    if line2:
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(W/2, by+bh-36, line1)
        c.setFont("Helvetica-Bold", 22)
        c.drawCentredString(W/2, by+bh-66, line2)
    else:
        c.setFont("Helvetica-Bold", 26)
        c.drawCentredString(W/2, by+bh-48, line1)

    # dots
    for i in range(total):
        dx = W/2-(total-1)*8+i*16
        if i==num-1:
            c.setFillColorRGB(*acc)
            c.circle(dx, 26, 6, fill=1, stroke=0)
        else:
            c.setFillColorRGB(*bg)
            c.setStrokeColorRGB(*acc)
            c.setLineWidth(1.5)
            c.circle(dx, 26, 4, fill=1, stroke=1)

def make_pdf(path):
    c = canvas.Canvas(path, pagesize=A4)
    draw_cover(c); c.showPage()
    for i,page in enumerate(PAGES):
        draw_page(c, page, i+1, len(PAGES))
        c.showPage()
    c.save()
    print(f"Saved: {path}")

if __name__ == "__main__":
    make_pdf("/home/user/claude/zoraiz_libro_v2.pdf")
