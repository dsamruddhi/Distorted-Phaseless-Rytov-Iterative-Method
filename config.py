import numpy as np
from scipy.special import jv as bessel1
from scipy.special import hankel1

""" System parameters """
frequency = 2e9
wavelength = 3e8 / frequency
wave_number = 2 * np.pi / wavelength
impedance = 120 * np.pi

""" DoI parameters """
doi_length = wavelength
m = 120
grid_length = doi_length / m
num_grids = m ** 2

""" Grid positions/ centroids """
centroids_x = np.arange(start=- doi_length / 2 + grid_length / 2, stop=doi_length / 2, step=grid_length)
centroids_y = np.arange(start=doi_length / 2 - grid_length / 2, stop=-doi_length / 2, step=-grid_length)
grid_positions = np.meshgrid(centroids_x, centroids_y)
# Grid radius
grid_radius = np.sqrt(grid_length ** 2 / np.pi)

""" Sensor parameters """
geometry = "circle"
rx_count = 40
tx_count = 40

""" Sensor positions """
x = 5 * wavelength
y = x
angles = np.arange(start=0, stop=360, step=360/rx_count)
theta = [np.deg2rad(angle) for angle in angles]
sensor_x = x*np.cos(theta)
sensor_y = y*np.sin(theta)
sensor_positions = np.transpose(np.array([sensor_x, sensor_y]))

""" Sensor links """
sensor_links = []
for i in range(rx_count):
    for j in range(rx_count):
        if i != j:
            sensor_links.append((i, j))

""" Other parameters """
nan_remove = True
noise_level = 0

""" Other constants to be used in code """
C1 = -impedance * np.pi * (grid_radius / 2)
C2 = bessel1(1, wave_number * grid_radius)
C3 = hankel1(1, wave_number * grid_radius)