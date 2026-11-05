# Genera pipeline.png, siguiendo el esquema visto en clase
import matplotlib.pyplot as plt

fig, eje = plt.subplots(figsize=(17, 9))
eje.set_xlim(0, 17)
eje.set_ylim(0, 9.4)
eje.axis("off")

AZUL, RELLENO, VERDE, RELLENO_V = "#2b5d8c", "#dbe9f6", "#2e7d4f", "#dcefe2"


def caja(x, y, w, h, texto, tam=9.5, negrita=False, borde=AZUL, relleno=RELLENO):
    eje.add_patch(plt.Rectangle((x, y), w, h, facecolor=relleno, edgecolor=borde, linewidth=1.5))
    eje.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=tam,
             fontweight="bold" if negrita else "normal")


def etiqueta(x, y, t, tam=13):
    eje.text(x, y, t, ha="center", va="center", fontsize=tam, fontweight="bold")


def flecha(p1, p2, color=AZUL):
    eje.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def linea(puntos, color=AZUL):
    xs, ys = zip(*puntos)
    eje.plot(xs, ys, color=color, lw=1.5)
    flecha(puntos[-2], puntos[-1], color)


Y, Y2, Y3 = 5.3, 2.9, 0.9

# caja punteada del pipeline de sklearn
eje.add_patch(plt.Rectangle((7.2, 3.95), 6.3, 2.95, facecolor="none", edgecolor=AZUL, linewidth=2, linestyle="--"))
eje.text(7.35, 6.65, "Pipeline", fontsize=13, fontweight="bold", color=AZUL, va="center")

# dataset -> EDA -> split -> X, y
caja(0.2, Y - 0.7, 1.8, 1.4, "dataset\nfutbolistas_2024_25.csv\n6764 x 17", tam=7.5)
flecha((2.0, Y), (2.3, Y))
caja(2.3, Y - 0.7, 1.8, 1.4, "EDA", tam=11)
eje.text(2.3, Y - 0.85, "- variables: tipo, distribucion,\n  sesgo, outliers\n- duplicados: 0\n- faltantes: valor, altura, pie\n- quitar menos de 90 minutos\n- hipotesis: liga y edad pesan mas",
         fontsize=8.5, va="top")
flecha((4.1, Y), (4.5, Y))
caja(4.5, Y - 0.7, 2.0, 1.4, "transformaciones\ncaro, edad2, log1p\ntrain-test split 70 / 30\nrandom_state = 42", tam=8.5)
flecha((6.5, Y + 0.3), (7.4, Y + 0.3))
etiqueta(6.9, Y + 0.65, "X")
etiqueta(6.9, Y - 0.72, "y")
eje.plot([6.5, 7.3], [Y - 0.3, Y - 0.3], color=AZUL, lw=1.5)

# ingenieria de caracteristicas -> X* -> fit
caja(7.4, Y - 0.8, 2.8, 1.6, "ing. de caracteristicas\nColumnTransformer\nStandardScaler (10 numericas)\nOneHotEncoder (3 categoricas)", tam=8.5)
flecha((10.2, Y + 0.3), (10.9, Y + 0.3))
etiqueta(10.55, Y + 0.65, "X*")
caja(10.9, Y - 0.8, 2.4, 1.6, "fit\nLogisticRegression\nSVC (kernel RBF)\nMLPClassifier", tam=9)
linea([(7.3, Y - 0.3), (7.3, 4.15), (10.6, 4.15), (10.6, Y - 0.3), (10.9, Y - 0.3)])

# ciclo de optimizacion bayesiana
caja(7.4, 7.5, 2.8, 1.5, "K-Fold con k = 5 sobre train\nAUC promedio de H", tam=9, borde=VERDE, relleno=RELLENO_V)
caja(10.9, 7.5, 2.4, 1.5, "proceso gaussiano\n" + r"$H_{sig} = \arg\max\ \mu + 2\sigma$" + "\n30 evaluaciones", tam=9, borde=VERDE, relleno=RELLENO_V)
flecha((10.2, 8.25), (10.9, 8.25), VERDE)
etiqueta(10.55, 8.6, "AUC", 10)
linea([(12.1, 7.5), (12.1, Y + 0.8)], VERDE)
etiqueta(12.45, 7.2, "H", 12)
linea([(8.8, 6.9), (8.8, 7.5)], VERDE)
eje.text(7.0, 8.25, "optimizacion\nbayesiana", fontsize=10.5, fontweight="bold", color=VERDE, ha="right", va="center")

# modelo entrenado
flecha((13.3, Y), (14.0, Y))
caja(14.0, Y - 0.7, 2.3, 1.4, "modelo entrenado\ncon el mejor H", negrita=True)

# predict y evaluacion
linea([(15.15, Y - 0.7), (15.15, 3.7), (9.3, 3.7), (9.3, Y2 + 0.5)])
caja(8.6, Y2 - 0.5, 1.4, 1.0, "predict")
flecha((10.0, Y2), (10.3, Y2))
etiqueta(10.6, Y2 + 0.05, r"$\hat{y}_{new}$")
flecha((10.9, Y2), (11.2, Y2))
caja(11.2, Y2 - 0.75, 2.6, 1.5, "evaluacion\nAUC media y desviacion (k = 10)\nAUC, recall, precision en test\ncurva ROC, matriz de confusion", tam=8.5)
flecha((13.8, Y2), (14.2, Y2))
caja(14.2, Y2 - 0.75, 2.1, 1.5, "interpretacion\ncoeficientes y casos\nque no se explican", tam=8.5)

# rama de datos nuevos
linea([(5.5, Y - 0.7), (5.5, Y3), (6.1, Y3)])
etiqueta(6.75, Y3, r"$X_{new}$" + "\n(X_test)", 10.5)
flecha((7.2, Y3), (7.5, Y3))
caja(7.5, Y3 - 0.55, 2.6, 1.1, "ing. de caracteristicas\n(el mismo preprocesador\nya ajustado con train)", tam=8.5)
flecha((10.1, Y3), (10.4, Y3))
etiqueta(10.85, Y3 + 0.05, r"$X^*_{new}$", 12)
linea([(11.3, Y3), (11.7, Y3), (11.7, 1.95), (9.3, 1.95), (9.3, Y2 - 0.5)])

plt.savefig("pipeline.png", dpi=100, bbox_inches="tight")
