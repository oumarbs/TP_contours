# TP contours : Sobel, Laplace, Canny

Comparaison de 3 détecteurs de contours avec OpenCV (C++) sur la base d'images de l'université de Groningen
(40 images 512x512 et leurs contours de référence tracés à la main).

**Vidéos explicatives (2 parties) :** https://1drv.ms/f/c/18aa5be2da13fabc/IgAIzMLzP1mtSIZir8p9cgWQAYSw0MRjaYpoPDuASovtNqs?e=MEMSkd


- `main.cpp` : code source
- `Rapport_TP_contours_Sall_Oumar.pdf` : rapport
- `images/` : images de la base, références dans `images/gt/`
- `resultats/` : fichiers CSV et images de comparaison (vert = correct, rouge = faux positif, bleu = faux négatif)

## Compilation

```
g++ -std=c++17 -O2 main.cpp -o tp_contours $(pkg-config --cflags --libs opencv4)
```

Sous Windows avec WSL : `run.bat` (ou Ctrl+Shift+B dans VSCode).

## Utilisation

Mode interactif (trackbars : image, détecteur, flou, seuils, noyau, voisinage) :

```
./tp_contours
```

Touches : `l` = appliquer à tout le lot, `s` = sauvegarder, `q` = quitter. Détecteur : 0 = Sobel, 1 = Laplace, 2 = Canny.

Traitement d'un lot d'images :

```
./tp_contours lot <sobel|laplace|canny> <flou> <seuil> [seuil_haut] [voisinage]
./tp_contours lot sobel 4 80
./tp_contours lot laplace 6 16
./tp_contours lot canny 4 40 80
```

Résultats dans `resultats/lot_<detecteur>.csv` : contours détectés, référence, corrects, faux positifs, faux négatifs,
P, TFP, TFN, précision, rappel et F.

Recherche des meilleurs paramètres :

```
./tp_contours balayage
```

## Résultats (voisinage 3x3, moyenne sur 40 images)

| Détecteur | Paramètres | P | TFP | TFN | F |
|---|---|---|---|---|---|
| Sobel | flou 4, seuil 80 | 0.398 | 0.360 | 0.243 | 0.471 |
| Laplace | flou 6, seuil 16 | 0.227 | 0.481 | 0.292 | 0.378 |
| Canny | flou 4, bas 40, haut 80 | 0.330 | 0.301 | 0.369 | 0.496 |

Base d'images : http://www.cs.rug.nl/~imaging/databases/contour_database/contour_database.html
