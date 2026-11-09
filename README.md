# TP2 – Procesamiento de Imágenes

**Integrantes:** Victoria Marull, Inés Medina y Paz Sevilla 

Trabajo Práctico N° 2: detección y clasificación de monedas y dados (Problema 1) y detección de patentes con segmentación de caracteres (Problema 2).

---

## Estructura del repositorio

```
tp2/
├── problema1_monedas.py      # Problema 1: monedas y dados
├── problema2_patentes.py     # Problema 2: patentes
├── monedas.jpg               # Imagen del Problema 1
├── Patentes/
│   └── Patentes/
│       └── img01.png ... img12.png   # Imágenes del Problema 2
└── README.md
```

## Requisitos

- Python 3
- Librerías: OpenCV, NumPy y Matplotlib

```bash
pip install opencv-python numpy matplotlib
```

## Cómo ejecutar

Desde la carpeta `tp2`:

```bash
python problema1_monedas.py
python problema2_patentes.py
```

Cada script muestra en ventanas los resultados de **cada etapa** del procesamiento e imprime en la terminal los valores obtenidos. Para avanzar a la siguiente etapa, cerrar la ventana actual.

---

## Problema 1 – Monedas y dados

### A. Segmentación

El fondo de la imagen **no es uniforme** (más oscuro a la izquierda que a la derecha), por lo que un umbral global como Otsu no separa bien los objetos. Por eso se segmenta **por bordes**:

1. Escala de grises y **suavizado Gaussiano** (9×9) para reducir el ruido.
2. **Canny** (umbrales 30 y 90) para detectar los bordes de los objetos, independientemente del brillo del fondo.
3. **Dilatación** (elemento elíptico 41×41) para cerrar los cortes de los contornos.
4. **Rellenado de huecos** por reconstrucción morfológica.
5. **Erosión** con el mismo elemento, para devolver a los objetos su tamaño real.
6. **Apertura** (elíptica 71×71) para alisar los bordes y eliminar irregularidades.
7. **Componentes conectadas** para separar cada objeto: se obtienen **19 objetos**.

### B. Clasificación de monedas

- **Factor de forma** ρ = 4π·Área / Perímetro²: vale 1 para un círculo y ≈ 0,785 para un cuadrado.
  - ρ > 0,85 → **moneda** (valores obtenidos: 0,88 – 0,90)
  - ρ ≤ 0,85 → **dado** (valores obtenidos: 0,76 y 0,80)
- **Área** de cada moneda para identificar su tipo:

| Tipo | Área (px) | Cantidad |
|---|---|---|
| 10 centavos | < 75.000 | 9 |
| 1 peso | 75.000 – 93.000 | 5 |
| 50 centavos | > 93.000 | 3 |

**Resultado:** 17 monedas, por un total de **$7,40**.

### C. Valor de los dados

Para cada dado:
1. Se recorta su región y se pinta de blanco todo lo que queda fuera de su máscara.
2. **Otsu invertido** dentro del recorte: localmente solo hay dos tonos (dado blanco y puntos negros), por lo que un umbral único sí funciona.
3. **Componentes conectadas** sobre los puntos.
4. Se cuentan solo los puntos de la **cara superior**: área mínima, para descartar ruido, y **factor de forma > 0,8**. Los puntos de las caras laterales se ven aplastados por la perspectiva y quedan descartados.

**Resultado:** dado 1 = **5**, dado 2 = **3**.

---

## Problema 2 – Patentes

### A. Detección de la patente

En lugar de buscar el rectángulo de la patente (en la imagen hay rejillas, faros y paragolpes con formas parecidas), se buscan **los caracteres**, y la patente es el lugar donde aparecen juntos.

1. **Umbralado:** los caracteres son blancos sobre fondo negro.
2. **Componentes conectadas** y filtros:
   - **Área** entre 15 y 200 px.
   - **Relación de aspecto** (alto/ancho) entre 1,5 y 3,5 y altura entre 8 y 25 px: un caracter es más alto que ancho.
3. **Dilatación horizontal** (rectángulo 15×3): los caracteres cercanos se unen en un solo objeto, y las manchas aisladas quedan solas.
4. **Componentes conectadas** sobre esa imagen: la patente es el objeto con **relación ancho/alto entre 2 y 6** y con **al menos 5 caracteres adentro**.
5. **Barrido de umbrales:** no existe un único umbral que funcione para las 12 imágenes (algunas necesitan 145 y otras 75). El algoritmo prueba umbrales de 60 a 200 y se queda con el primero que encuentra una patente con 6 caracteres (o 5, si ninguno llega a 6).

### B. Segmentación de caracteres

1. Se **recorta la patente** con un margen de una letra a cada lado.
2. Nuevo **barrido de umbrales dentro del recorte**, hasta encontrar exactamente 6 componentes con forma de caracter.
3. Si un caracter resulta más alto de lo normal (porque quedó pegado a un pedazo del marco de la patente), se **recorta a la altura de su caracter vecino**.
4. Se **ordenan de izquierda a derecha**.

**Caso particular (img03):** la "D" toca el marco blanco de la patente. En la Parte A se pierde, pero alcanza con las otras 5 letras para ubicar la patente. En la Parte B, al umbralizar solo dentro del recorte, la "D" se separa del marco y se recupera.

**Resultado:** patente detectada y **6 caracteres segmentados en las 12 imágenes**.

### Visualización

- Una figura por imagen con las 10 etapas del procesamiento.
- Un resumen con las 12 patentes recortadas y sus caracteres marcados.
- Una figura con el recorte de cada patente y su máscara de caracteres.
