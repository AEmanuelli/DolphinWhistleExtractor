# import cv2
# import glob
# import numpy as np

# def load_matplotlib_images(folder_path):
#     """
#     Charge toutes les images JPG depuis le dossier spécifié.
    
#     Paramètres:
#         folder_path (str): Chemin vers le dossier contenant les images.
    
#     Retourne:
#         images (list): Liste d'images chargées sous forme de tableaux NumPy.
#     """
#     # Recherche tous les fichiers .jpg dans le dossier
#     image_paths = glob.glob(folder_path + "/*.jpg")
#     images = []
#     for path in image_paths:
#         # cv2.imread lit l'image en format BGR
#         img = cv2.imread(path)
#         if img is not None:
#             images.append(img)
#         else:
#             print(f"Impossible de charger l'image: {path}")
#     return images

# def transform_image_format(image, target_width_px=903, target_height_px=677):
#     """
#     Transforme une image générée avec matplotlib (format RGB ou BGR) en une image au format
#     utilisé par process_audio_file_super_fast (grayscale normalisé et converti en BGR).
    
#     Paramètres:
#         image (np.array): Image générée par process_audio_file.
#         target_width_px (int): Largeur cible en pixels.
#         target_height_px (int): Hauteur cible en pixels.
    
#     Retourne:
#         transformed (np.array): Image transformée au format BGR.
#     """
#     # Si l'image est en BGR (comme chargée par cv2), on peut la convertir en RGB pour être sûr
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Convertir l'image de RGB à grayscale
#     gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
#     # Redimensionner l'image aux dimensions cibles
#     resized = cv2.resize((target_width_px, target_height_px), interpolation=cv2.INTER_LINEAR)
    
#     # Convertir en image BGR (3 canaux)
#     transformed = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    
#     return transformed

# # # Exemple d'utilisation :
# # folder = "./images"  # chemin du dossier où sont sauvegardées les images générées avec matplotlib
# # matplotlib_images = load_matplotlib_images(folder)


# # # Affichage de la première image transformée pour vérification (si vous utilisez un environnement Jupyter)
# # import matplotlib.pyplot as plt

# # if transformed_images:
# #     plt.imshow(cv2.cvtColor(transformed_images[0], cv2.COLOR_BGR2RGB))
# #     plt.axis('off')
# #     plt.show()



# images = load_matplotlib_images("/home/emanuelli/Documents/GitHub/Dolphins/speed_test5/20241129_154401/positive")
# # Transformation de chaque image
# transformed_images = [transform_image_format(img) for img in images]
# # Affichage de la première image transformée pour vérification (si vous utilisez un environnement Jupyter)
# #save the figure in the folder
# for i, img in enumerate(transformed_images):
#     cv2.imwrite(f"image_{i}_convert?.jpg", img)
#     print(f"image_{i}.jpg saved")

from matplotlib import colormaps
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# print the gray colormaps
list = [cmap for cmap in colormaps() if  cmap.lower()=="gray"]

cmaps = {}

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list


plot_color_gradients('Perceptually Uniform Sequential', list)
# plt.show()
plt.savefig("gray_colormaps.png")
print("gray colormaps saved")
# print(cmaps