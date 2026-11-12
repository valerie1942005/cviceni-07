from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
import skimage

plt.close('all')

# =============================================================================
# Ukol 1
# =============================================================================
# Nactete obraz 'texture.png' a prevedte jej na sedotonovy. Zobrazte si jej i s jeho histogramem.
# Transformujte originalni obraz na parametricky obraz:
#       Připravte novou prázdnou výstupní proměnnou o stejné velikosti jako je vstupní obraz a s datovým typem float.
#       Pomocí dvou vnořených for cyklů projděte celý obraz a vybírejte submatice o velikosti 9x9 px. Ze submatic vypočtěte parametr:
#           Směrodatné odchylky a uložte na odpovídající pozici ve výstupní matici
#           Rozsahu pomocí nelineárního rozsahového filtru, jehož principem je nalezení lokálního rozdílu mezi maximem a minimem v submatici.
# Zobrazte parametrický obraz a jeho histogram
# =============================================================================
texture_img = imread("data/texture.png")
grayscale_texture_img = rgb2gray(texture_img)

# definice velikosti obrazku a submatice:
[rows, cols] = np.shape(grayscale_texture_img)
size_submatice = 9
size_pul_submatice = round(size_submatice/2)

# vykresleni puvodniho obrazku a histogramu
plt.figure()
plt.subplot(2,2,1)
plt.imshow(grayscale_texture_img, cmap='gray')
plt.axis('off')
plt.subplot(2,2,2)
plt.hist(grayscale_texture_img.ravel(), 256, range=[0, 1])
plt.xlabel('Jas [-]')
plt.ylabel('Cetnost [-]')
plt.show()

# definice parametrickych map
param_space_std = np.zeros_like(grayscale_texture_img, dtype=float)
param_space_rozsah = np.zeros_like(grayscale_texture_img, dtype=float)

# definice vyuzitelne oblasti
area_rows = np.linspace(size_pul_submatice,rows-size_pul_submatice, rows-size_submatice+1, dtype = int) #start, stop, pocet
area_cols = np.linspace(size_pul_submatice,cols-size_pul_submatice, cols-size_submatice+1, dtype = int)

# vypocet parametru do parametrickych map:
for row in area_rows:
    for col in area_cols:
        local_area = grayscale_texture_img[row - size_pul_submatice:row + size_pul_submatice + 1, col - size_pul_submatice:col + size_pul_submatice + 1]
        param_space_std[row,col] = np.std(local_area)
        param_space_rozsah[row,col] = np.max(local_area) - np.min(local_area)

# param_space_std = param_space_std[size_pul_submatice:rows-size_pul_submatice, size_pul_submatice:cols-size_pul_submatice]
# param_space_rozsah = param_space_rozsah[size_pul_submatice:rows-size_pul_submatice, size_pul_submatice:cols-size_pul_submatice]

# vykresleni:
plt.figure()
plt.subplot(2,3,1)
plt.imshow(grayscale_texture_img, cmap = 'gray')
plt.axis('off')
plt.title('Original')
plt.subplot(2,3,4)
plt.hist(grayscale_texture_img.ravel(), bins = 256)
plt.xlabel('Jas [-]')
plt.ylabel('Cetnost [-]')
plt.subplot(2,3,2)
plt.imshow(param_space_std[:,:], cmap = 'gray')
plt.axis('off')
plt.title('Parametricky prostor - std')
plt.subplot(2,3,5)
plt.hist(param_space_std[:,:].ravel(), bins = 256)
plt.xlabel('Jas [-]')
plt.ylabel('Cetnost [-]')
plt.subplot(2,3,3)
plt.imshow(param_space_rozsah[:,:], cmap = 'gray')
plt.axis('off')
plt.title('Parametricky prostor - rozsah')
plt.subplot(2,3,6)
plt.hist(param_space_rozsah[:,:].ravel(), bins = 256)
plt.xlabel('Jas [-]')
plt.ylabel('Cetnost [-]')
plt.show()


# =============================================================================
# Ukol 2
# =============================================================================
# Nactete obraz ‘monsters.png’ a prevedte jej na sedotonovy a zobrazte.
# Vytvorte hrubou hranovou reprezentaci s vyuzitim první parcialni derivace:
#   Vytvorte gradientní obraz
#       Vypoctete prvni parcialni derivaci pomoci Sobelovy masky ve smeru x a y
#       Vytvorte gradientni obraz pomoci vypoctu euklidovske vzdalenosti
#   Gradientni obraz vhodne naprahujte a zobrazte
# =============================================================================
# sobeluv operator:
sobel_maska_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
sobel_maska_y = sobel_maska_x.T

img_monsters = imread("data/monsters.png")
grayscale_img_monsters = rgb2gray(img_monsters)

# vytvořte "gradientní obraz"
gradient_x = convolve2d(grayscale_img_monsters, sobel_maska_x, mode='same')
gradient_y = convolve2d(grayscale_img_monsters, sobel_maska_y, mode='same')

gradient_obraz = np.hypot(gradient_x, gradient_y)

# naprahujte "gradientni obraz":
prah_gradient = 0.3
gradient_obraz_prah = np.copy(gradient_obraz)
gradient_obraz_prah[gradient_obraz_prah < prah_gradient] = 0
gradient_obraz_prah[gradient_obraz_prah >= prah_gradient] = 1

# vykresleni:
plt.figure()
plt.subplot(2,3,1)
plt.imshow(grayscale_img_monsters, cmap = 'gray')
plt.axis('off')
plt.subplot(2,3,2)
plt.imshow(gradient_x, cmap = 'gray')
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(gradient_y, cmap = 'gray')
plt.axis('off')
plt.subplot(2,3,4)
plt.imshow(gradient_obraz, cmap = 'gray')
plt.axis('off')
plt.subplot(2,3,5)
plt.imshow(gradient_obraz, cmap = 'gray')
plt.axis('off')
plt.subplot(2,3,6)
plt.imshow(gradient_obraz_prah, cmap = 'gray')
plt.axis('off')
plt.show()


# =============================================================================
# Ukol 3
# =============================================================================
# Vytvorte hrubou hranovou reprezentaci obrazu monsters s vyuzitim Laplacianu obrazu:
#   Prozkoumejte prilozenou funkci 'zero_crossing_detection()'
#   S vyuzitim funkce 'zero_crossing_detection()' vytvorte hrubou hranovou reprezentaci z Laplacianu
# =============================================================================

def zero_crossing_detection(Laplacian_obraz, threshold):
    RER2 = np.zeros([Laplacian_obraz.shape[0], Laplacian_obraz.shape[1]])
    M = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    [rows, cols] = np.shape(Laplacian_obraz)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            subimage = Laplacian_obraz[i - 1:i + 2, j - 1:j + 2]
            aux_var = (subimage * M).ravel()
            if (np.sum(aux_var) == 0):
                continue
            aux_var = np.delete(aux_var, np.where(aux_var == 0))
            if len(aux_var) == 0:
                continue
            maximum = np.max(aux_var)
            minimum = np.min(aux_var)
            if minimum < 0 and maximum > 0:  # 1st condition
                diference = maximum - minimum
                if diference > threshold:  # 2nd condition
                    value = Laplacian_obraz[i, j]
                    if value < maximum and value > minimum:  # 3rd condition
                        RER2[i, j] = 1
    return (RER2)

Laplac_operator = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
Laplacian_obraz = convolve2d(grayscale_img_monsters, Laplac_operator, mode='same')

prah_laplacian = 0.05
RER = zero_crossing_detection(Laplacian_obraz, prah_laplacian)

# vykresleni:
plt.figure()
plt.subplot(1,3,1)
plt.title("Původní obraz")
plt.imshow(grayscale_img_monsters, cmap = 'gray')
plt.axis('off')
plt.subplot(1,3,2)
plt.title("Laplacian obraz")
plt.imshow(Laplacian_obraz, cmap = 'gray')
plt.axis('off')
plt.subplot(1,3,3)
plt.title("HHR - detekce průchodu nulou")
plt.imshow(RER, cmap = 'gray')
plt.axis('off')
plt.show()

plt.figure()
plt.subplot(1,3,1)
plt.title("Původní obraz")
plt.imshow(grayscale_img_monsters, cmap = 'gray')
plt.axis('off')
plt.subplot(1,3,2)
plt.title("HHR - prah. gradientni obraz")
plt.imshow(gradient_obraz_prah, cmap = 'gray')
plt.axis('off')
plt.subplot(1,3,3)
plt.title("HHR - detekce průchodu nulou")
plt.imshow(RER, cmap = 'gray')
plt.axis('off')
plt.show()