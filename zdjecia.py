import cv2
import matplotlib.pyplot

sciezka = "1.jpg"

obraz = cv2.imread(sciezka, cv2.IMREAD_COLOR)

if obraz is None:
    raise ValueError(f"Nie udało się wczytać obrazu z lokalizacji: {sciezka}")

print("Oryginalny rozmiar:", obraz.shape)

obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)

matplotlib.pyplot.figure(figsize=(6, 4))
matplotlib.pyplot.imshow(obraz_rgb)
matplotlib.pyplot.title("Obraz oryginalny")
matplotlib.pyplot.axis("off")
matplotlib.pyplot.show()

wysokosc, szerokosc = obraz.shape[:2]

obraz_mniejszy = cv2.resize(
    obraz,
    (max(1, szerokosc // 2), max(1, wysokosc // 2)),
    interpolation=cv2.INTER_AREA
)

print("Rozmiar po zmniejszeniu:", obraz_mniejszy.shape)

obraz_szary = cv2.cvtColor(
    obraz_mniejszy,
    cv2.COLOR_BGR2GRAY
)

obraz_wynikowy = cv2.rotate(
    obraz_szary,
    cv2.ROTATE_90_CLOCKWISE
)

matplotlib.pyplot.figure(figsize=(6, 4))
matplotlib.pyplot.imshow(obraz_wynikowy, cmap="gray", vmin=0, vmax=255)
matplotlib.pyplot.title("Obraz po zmniejszeniu, konwersji i obrocie")
matplotlib.pyplot.axis("off")
matplotlib.pyplot.show()

print("\nMacierz obrazu wynikowego:")
print(obraz_wynikowy)

print("\nWymiary macierzy:", obraz_wynikowy.shape)
print("Typ danych:", obraz_wynikowy.dtype)

print("\nFragment macierzy 10 x 10:")
print(obraz_wynikowy[:10, :10])