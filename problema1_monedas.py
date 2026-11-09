import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Función auxiliar para mostrar imágenes ----------
def imshow(img, title=None, color_img=False):
    plt.figure(figsize=(10, 7))
    if color_img:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# ---------- Paso 1: cargar la imagen ----------
img = cv2.imread('monedas.jpg')
print('Tamaño de la imagen (alto, ancho, canales):', img.shape)
imshow(img, 'Imagen original', color_img=True)

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow(gris, 'Escala de grises')

# ---------- Paso 2a: intento con umbral global (Otsu) ----------
# Lo dejamos para mostrar por qué NO sirve con fondo no uniforme
_, otsu = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow(otsu, 'Umbral de Otsu: falla por el fondo no uniforme')

# ---------- Paso 2b: suavizado + bordes con Canny ----------
suave = cv2.GaussianBlur(gris, (9, 9), 0)
imshow(suave, 'Suavizado Gaussiano 9x9')

bordes = cv2.Canny(suave, 30, 90) #si es menor a 40 se descarta
imshow(bordes, 'Bordes con Canny')

# ---------- Funciones de la U6: reconstrucción y rellenado de huecos ----------
def imreconstruct(marker, mask, kernel=np.ones((3, 3), np.uint8)):
    while True:
        expanded = cv2.dilate(marker, kernel)
        expanded = cv2.bitwise_and(expanded, mask)
        if (marker == expanded).all():
            return expanded
        marker = expanded

def imfillhole(img):
    mask = np.zeros_like(img)
    mask = cv2.copyMakeBorder(mask[1:-1, 1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    marker = cv2.bitwise_not(img, mask=mask)
    img_c = cv2.bitwise_not(img)
    img_r = imreconstruct(marker, img_c)
    return cv2.bitwise_not(img_r)

# ---------- Paso 3a: dilatar los bordes para cerrar cortes ----------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
bordes_dil = cv2.dilate(bordes, kernel)
imshow(bordes_dil, 'Bordes dilatados (kernel elíptico 55x55)')

# ---------- Paso 3b: rellenar huecos ----------
relleno = imfillhole(bordes_dil)
imshow(relleno, 'Huecos rellenos')

# ---------- Paso 3c: erosionar para devolver el tamaño real ----------
mascara = cv2.erode(relleno, kernel)
imshow(mascara, 'Máscara final (erosionada)')

# ---------- Paso 3c bis: apertura para alisar bordes y limpiar ----------
kernel_ap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_ap)
imshow(mascara, 'Apertura (bordes alisados)')

# ---------- Paso 3d: contar objetos ----------
n, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(mascara)
print('Cantidad de objetos detectados:', n - 1)

# ---------- Paso 4: factor de forma para separar monedas de dados ----------
UMBRAL_RHO = 0.85

monedas = []   # guardamos la etiqueta y el área de cada moneda
dados = []     # guardamos la etiqueta de cada dado
resultado = img.copy()

print('\nObjeto |   Área   | Factor de forma | Tipo')
for i in range(1, n):   # arrancamos en 1 porque 0 es el fondo
    obj = (etiquetas == i).astype(np.uint8)
    contornos, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = stats[i, cv2.CC_STAT_AREA]
    perimetro = cv2.arcLength(contornos[0], True)
    rho = 4 * np.pi * area / perimetro**2

    if rho > UMBRAL_RHO:
        tipo = 'moneda'
        monedas.append((i, area))
        color = (255, 0, 0)     # azul (BGR)
    else:
        tipo = 'dado'
        dados.append(i)
        color = (0, 0, 255)     # rojo (BGR)

    print(f'{i:6d} | {area:8d} | {rho:15.3f} | {tipo}')

    x, y, w, h = stats[i, :4]
    cv2.rectangle(resultado, (x, y), (x + w, y + h), color, 10)
    cv2.putText(resultado, str(i), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 8)

print(f'\nTotal: {len(monedas)} monedas y {len(dados)} dados')
imshow(resultado, 'Monedas (azul) y dados (rojo)', color_img=True)

# ---------- Paso 5: clasificar monedas por área ----------
UMBRAL_10C = 75000     # menor a esto: 10 centavos
UMBRAL_1P = 93000      # entre los dos umbrales: 1 peso; mayor: 50 centavos

conteo = {'10 centavos': 0, '1 peso': 0, '50 centavos': 0}
colores = {'10 centavos': (255, 200, 0),    # celeste (BGR)
           '1 peso': (180, 105, 255),       # rosa (BGR)
           '50 centavos': (0, 200, 0)}      # verde (BGR)
clasificadas = img.copy()

print('\nObjeto |   Área   | Tipo de moneda')
for i, area in monedas:
    if area < UMBRAL_10C:
        tipo = '10 centavos'
    elif area < UMBRAL_1P:
        tipo = '1 peso'
    else:
        tipo = '50 centavos'
    conteo[tipo] += 1
    print(f'{i:6d} | {area:8d} | {tipo}')

    x, y, w, h = stats[i, :4]
    cv2.rectangle(clasificadas, (x, y), (x + w, y + h), colores[tipo], 12)
    cv2.putText(clasificadas, tipo, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.5, colores[tipo], 8)

print('\nConteo de monedas:')
for tipo, cant in conteo.items():
    print(f'  {tipo}: {cant}')
total = conteo['10 centavos'] * 0.10 + conteo['1 peso'] * 1 + conteo['50 centavos'] * 0.50
print(f'  Total: {sum(conteo.values())} monedas = ${total:.2f}')

imshow(clasificadas, 'Clasificación de monedas', color_img=True)

# ---------- Paso 6: contar los puntos de la cara superior de cada dado ----------
AREA_MIN_PUNTO = 1000
RHO_MIN_PUNTO = 0.8

resultado_dados = img.copy()
suma_dados = 0

for d, i in enumerate(dados, start=1):
    x, y, w, h = stats[i, :4]

    # 6a: recortar el dado y blanquear lo que está fuera de su máscara
    recorte = gris[y:y + h, x:x + w].copy()
    mascara_dado = (etiquetas[y:y + h, x:x + w] == i)
    recorte[~mascara_dado] = 255
    imshow(recorte, f'Dado {d}: recorte')

    # 6b: Otsu invertido (puntos negros -> blancos)
    _, puntos_bin = cv2.threshold(recorte, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    imshow(puntos_bin, f'Dado {d}: umbral de Otsu invertido')

    # 6c: componentes conexas y filtro por área y factor de forma
    n_p, etiq_p, stats_p, _ = cv2.connectedComponentsWithStats(puntos_bin)
    valor = 0
    for j in range(1, n_p):
        area_p = stats_p[j, cv2.CC_STAT_AREA]
        if area_p < AREA_MIN_PUNTO:
            continue
        obj_p = (etiq_p == j).astype(np.uint8)
        cont_p, _ = cv2.findContours(obj_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rho_p = 4 * np.pi * area_p / cv2.arcLength(cont_p[0], True)**2
        if rho_p > RHO_MIN_PUNTO:
            valor += 1
            px, py, pw, ph = stats_p[j, :4]
            cv2.circle(resultado_dados, (x + px + pw // 2, y + py + ph // 2), pw // 2 + 10, (0, 0, 255), 8)

    print(f'Dado {d}: valor {valor}')
    suma_dados += valor
    cv2.rectangle(resultado_dados, (x, y), (x + w, y + h), (255, 0, 0), 10)
    cv2.putText(resultado_dados, f'Dado {d}: {valor}', (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 8)

print(f'Suma de los dados: {suma_dados}')
imshow(resultado_dados, 'Valor de cada dado', color_img=True)