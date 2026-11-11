import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, title=None, color_img=False):
    plt.figure(figsize=(10, 7))
    if color_img:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

RUTA = 'Patentes/Patentes/'   # carpeta donde están img01.png ... img12.png


# Funciones de la Parte A

def candidatos_letras(gris, umbral):
    """Pasos 1b a 1e: umbral, componentes conexas, filtro por área y por aspecto."""
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)

    filtro_area = np.zeros_like(binaria)
    filtro_aspecto = np.zeros_like(binaria)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 15 < area < 200:
            filtro_area[etiquetas == i] = 255
            if 1.5 < h / w < 3.5 and 8 <= h <= 25:
                filtro_aspecto[etiquetas == i] = 255
    return binaria, etiquetas, filtro_area, filtro_aspecto


def buscar_patente(etiquetas, filtro_aspecto):
    """Pasos 2a y 2b: dilatación horizontal y objeto con forma de patente."""
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    unidas = cv2.dilate(filtro_aspecto, kernel_h)
    n2, etiquetas2, stats2, _ = cv2.connectedComponentsWithStats(unidas)

    for j in range(1, n2):
        x, y, w, h, area = stats2[j]
        objeto = (etiquetas2 == j) & (filtro_aspecto > 0)
        letras_adentro = len(np.unique(etiquetas[objeto]))
        if 2 < w / h < 6 and letras_adentro >= 5:
            solo_letras = np.where(objeto, 255, 0).astype(np.uint8)
            return (x, y, w, h), letras_adentro, unidas, solo_letras
    return None, 0, unidas, None


def detectar_patente(gris):
    """Paso 3: barrido de umbrales. Devuelve el primer umbral que encuentra
    una patente con 6 letras; si ninguno llega a 6, el primero con 5."""
    umbral_5 = None
    for umbral in range(60, 200, 5):
        _, etiquetas, _, filtro_aspecto = candidatos_letras(gris, umbral)
        patente, n_letras, _, _ = buscar_patente(etiquetas, filtro_aspecto)
        if n_letras == 6:
            return umbral
        if n_letras == 5 and umbral_5 is None:
            umbral_5 = umbral
    return umbral_5



# Funciones de la Parte B


def segmentar_caracteres(gris, patente, solo_letras):
    """Paso 4: recorta la patente con margen, busca el umbral que da 6 caracteres
    y corrige los que salen más altos de lo normal (pedazo de marco)."""
    x, y, w, h = patente

    # altura típica de una letra (mediana de las letras halladas en la Parte A)
    _, _, stats_l, _ = cv2.connectedComponentsWithStats(solo_letras, connectivity=8)
    alto_letra = np.median(stats_l[1:, 3])

    # recortar la patente con margen (una letra a los costados)
    mx, my = int(alto_letra), int(0.4 * alto_letra)
    x0, y0 = max(0, x - mx), max(0, y - my)
    recorte = gris[y0:y + h + my, x0:x + w + mx]

    # barrido de umbrales dentro del recorte hasta encontrar 6 caracteres
    for umbral_b in range(60, 220, 5):
        _, bin_rec = cv2.threshold(recorte, umbral_b, 255, cv2.THRESH_BINARY)
        n_r, etiq_r, stats_r, _ = cv2.connectedComponentsWithStats(bin_rec, connectivity=8)
        caracteres = []
        for i in range(1, n_r):
            cx, cy, cw, ch, area = stats_r[i]
            if area > 15 and 0.7 * alto_letra <= ch <= 1.6 * alto_letra and 1.2 < ch / cw < 3.5:
                caracteres.append([cx, cy, cw, ch, i])
        if len(caracteres) == 6:
            break

    # ordenar de izquierda a derecha
    caracteres.sort(key=lambda c: c[0])

    # si un caracter es más alto de lo normal, recortarlo a la altura de su vecino
    for k, c in enumerate(caracteres):
        if c[3] > 1.2 * alto_letra:
            vecino = caracteres[k + 1] if k + 1 < len(caracteres) else caracteres[k - 1]
            arriba, abajo = vecino[1] - 1, vecino[1] + vecino[3] + 1
            mascara_c = (etiq_r == c[4]).astype(np.uint8)
            mascara_c[:arriba] = 0
            mascara_c[abajo:] = 0
            ys, xs = np.nonzero(mascara_c)
            c[:4] = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]

    #  máscara con solo los caracteres (cada uno limitado a su caja)
    mascara_car = np.zeros_like(recorte)
    for cx, cy, cw, ch, i in caracteres:
        caja = etiq_r[cy:cy + ch, cx:cx + cw] == i
        mascara_car[cy:cy + ch, cx:cx + cw][caja] = 255

    return recorte, bin_rec, umbral_b, caracteres, (x0, y0), alto_letra, mascara_car


# enderezar la patente (homografía)

def enderezar_patente(img, mascara_car, caracteres, origen, alto_letra):
    """Toma 4 esquinas a partir del primer y el último caracter, calcula la
    homografía que las lleva a un rectángulo y endereza la patente."""
    x0, y0 = origen
    m = int(0.4 * alto_letra)     # margen alrededor de los caracteres

    primero, ultimo = caracteres[0], caracteres[-1]
    # Esquinas en la imagen original: sup-izq, sup-der, inf-der, inf-izq
    origen_pts = np.float32([
        [x0 + primero[0] - m,             y0 + primero[1] - m],
        [x0 + ultimo[0] + ultimo[2] + m,  y0 + ultimo[1] - m],
        [x0 + ultimo[0] + ultimo[2] + m,  y0 + ultimo[1] + ultimo[3] + m],
        [x0 + primero[0] - m,             y0 + primero[1] + primero[3] + m],
    ])

    # Rectángulo destino: mismo ancho y alto que la patente, agrandado x4
    ESCALA = 4
    ancho = int(np.linalg.norm(origen_pts[1] - origen_pts[0])) * ESCALA
    alto = int(np.linalg.norm(origen_pts[3] - origen_pts[0])) * ESCALA
    destino_pts = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])

    H, _ = cv2.findHomography(origen_pts, destino_pts)
    patente_recta = cv2.warpPerspective(img, H, (ancho, alto))

    # Cajas de los caracteres: transformamos sus 4 esquinas con la misma H y las dibujamos en rojo
    for cx, cy, cw, ch, _ in caracteres:
        esquinas = np.float32([
            [x0 + cx,      y0 + cy],
            [x0 + cx + cw, y0 + cy],
            [x0 + cx + cw, y0 + cy + ch],
            [x0 + cx,      y0 + cy + ch],
        ]).reshape(-1, 1, 2)
        esquinas_rectas = cv2.perspectiveTransform(esquinas, H)
        cv2.polylines(patente_recta, [np.int32(esquinas_rectas)], True, (0, 0, 255), 2)

    # La máscara de caracteres está en coordenadas del recorte: la pasamos a la imagen completa
    mascara_img = np.zeros(img.shape[:2], np.uint8)
    mascara_img[y0:y0 + mascara_car.shape[0], x0:x0 + mascara_car.shape[1]] = mascara_car
    mascara_recta = cv2.warpPerspective(mascara_img, H, (ancho, alto), flags=cv2.INTER_NEAREST)

    return patente_recta, mascara_recta


# Procesar una imagen completa (Parte A + Parte B) y mostrar sus etapas

def procesar(nombre):
    img = cv2.imread(RUTA + nombre)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Parte A
    umbral = detectar_patente(gris)
    binaria, etiquetas, filtro_area, filtro_aspecto = candidatos_letras(gris, umbral)
    patente, n_letras, unidas, solo_letras = buscar_patente(etiquetas, filtro_aspecto)

    # Parte B
    recorte, bin_rec, umbral_b, caracteres, (x0, y0), alto_letra, mascara_car = segmentar_caracteres(gris, patente, solo_letras)

    # Caracteres sobre el recorte
    vis = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)
    for cx, cy, cw, ch, _ in caracteres:
        cv2.rectangle(vis, (cx, cy), (cx + cw - 1, cy + ch - 1), (0, 255, 0), 1)

    # patente final = caja que encierra los caracteres (coordenadas de la imagen original)
    px0 = x0 + min(c[0] for c in caracteres)
    py0 = y0 + min(c[1] for c in caracteres)
    px1 = x0 + max(c[0] + c[2] for c in caracteres)
    py1 = y0 + max(c[1] + c[3] for c in caracteres)
    margen = int(0.3 * alto_letra)
    final = img.copy()
    cv2.rectangle(final, (px0 - margen, py0 - margen), (px1 + margen, py1 + margen), (0, 255, 0), 2)
    for cx, cy, cw, ch, _ in caracteres:
        cv2.rectangle(final, (x0 + cx, y0 + cy), (x0 + cx + cw - 1, y0 + cy + ch - 1), (255, 0, 255), 1)

    print(f'{nombre}: umbral Parte A = {umbral} ({n_letras} letras) | '
          f'umbral Parte B = {umbral_b} | caracteres = {len(caracteres)}')

    # Todas las etapas de esta imagen en una sola figura
    etapas = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Original'),
        (binaria, f'A: umbral {umbral}'),
        (filtro_area, 'A: filtrado por área'),
        (filtro_aspecto, 'A: filtrado por aspecto'),
        (unidas, 'A: dilatación horizontal'),
        (solo_letras, 'A: letras que ubican la patente'),
        (recorte, 'B: recorte de la patente'),
        (bin_rec, f'B: recorte umbralizado ({umbral_b})'),
        (cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), 'B: caracteres segmentados'),
        (cv2.cvtColor(final, cv2.COLOR_BGR2RGB), 'Resultado final'),
    ]
    plt.figure(figsize=(18, 7))
    for k, (im, titulo) in enumerate(etapas, start=1):
        plt.subplot(2, 5, k)
        plt.imshow(im, cmap='gray')
        plt.title(titulo, fontsize=9)
        plt.axis('off')
    plt.suptitle(nombre)
    plt.tight_layout()
    plt.show()

    # Recorte para el resumen: patente con contexto, agrandada x4 y caracteres en rojo
    ESCALA = 4
    ancho_p, alto_p = px1 - px0, py1 - py0
    rx0, ry0 = max(0, px0 - int(0.3 * ancho_p)), max(0, py0 - int(1.0 * alto_p))
    rx1, ry1 = px1 + int(0.3 * ancho_p), py1 + int(1.0 * alto_p)
    resumen = cv2.resize(img[ry0:ry1, rx0:rx1], None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_CUBIC)
    for cx, cy, cw, ch, _ in caracteres:
        ax0 = (x0 + cx - rx0) * ESCALA
        ay0 = (y0 + cy - ry0) * ESCALA
        cv2.rectangle(resumen, (ax0, ay0), (ax0 + cw * ESCALA, ay0 + ch * ESCALA), (0, 0, 255), 2)

    # enderezar la patente con una homografía
    patente_recta, mascara_recta = enderezar_patente(img, mascara_car, caracteres, (x0, y0), alto_letra)

    return resumen, len(caracteres), vis, mascara_car, umbral_b, patente_recta, mascara_recta



# correr las 12 imágenes

resultados = []
for k in range(1, 13):
    nombre = f'img{k:02d}.png'
    resumen, n_car, vis, mascara_car, umbral_b, patente_recta, mascara_recta = procesar(nombre)
    resultados.append((nombre, resumen, n_car, vis, mascara_car, umbral_b, patente_recta, mascara_recta))

# Resumen: las 12 patentes recortadas con sus caracteres
detectadas = sum(1 for r in resultados if r[2] == 6)
plt.figure(figsize=(16, 9))
for k, (nombre, resumen, n_car, _, _, umbral_b, _, _) in enumerate(resultados, start=1):
    plt.subplot(3, 4, k)
    plt.imshow(cv2.cvtColor(resumen, cv2.COLOR_BGR2RGB))
    plt.title(f'{nombre[:-4]}: {n_car} caracteres (T={umbral_b})', fontsize=11)
    plt.axis('off')
plt.suptitle(f'Patentes detectadas: {detectadas} de {len(resultados)}', fontsize=13)
plt.tight_layout()
plt.show()

# Figura final: recorte de cada patente y su máscara de caracteres
plt.figure(figsize=(18, 10))
for k, (nombre, _, n_car, vis, mascara_car, _, _, _) in enumerate(resultados):
    plt.subplot(4, 6, 2 * k + 1)
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'{nombre}: recorte', fontsize=9)
    plt.axis('off')
    plt.subplot(4, 6, 2 * k + 2)
    plt.imshow(mascara_car, cmap='gray')
    plt.title(f'{nombre}: máscara ({n_car} caracteres)', fontsize=9)
    plt.axis('off')
plt.suptitle('Patentes recortadas y caracteres segmentados')
plt.tight_layout()
plt.show()

# Figura: patentes enderezadas con homografía y su máscara
plt.figure(figsize=(18, 10))
for k, (nombre, _, _, _, _, _, patente_recta, mascara_recta) in enumerate(resultados):
    plt.subplot(4, 6, 2 * k + 1)
    plt.imshow(cv2.cvtColor(patente_recta, cv2.COLOR_BGR2RGB))
    plt.title(f'{nombre}: enderezada', fontsize=9)
    plt.axis('off')
    plt.subplot(4, 6, 2 * k + 2)
    plt.imshow(mascara_recta, cmap='gray')
    plt.title(f'{nombre}: máscara enderezada', fontsize=9)
    plt.axis('off')
plt.suptitle('Patentes enderezadas con homografía (findHomography + warpPerspective)')
plt.tight_layout()
plt.show()