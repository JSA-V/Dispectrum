import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from utils.consonance_models import Dis2

def dissonance_surface(F, A, alpha_range, beta_range, method='Sethares1'):
    """
    Calcula y visualiza la superficie de disonancia para un espectro dado,
    variando las relaciones alpha y beta.
    
    Parámetros:
    - F: Vector de frecuencias base.
    - A: Vector de amplitudes base.
    - alpha_range: Rango de valores para alpha (segunda frecuencia variable / fija).
    - beta_range: Rango de valores para beta (primera frecuencia variable / segunda fija).
    - method: Modelo de disonancia ('Sethares1' o 'Vassilakis').
    """
    Alpha = np.linspace(*alpha_range, 50)
    Beta = np.linspace(*beta_range, 50)
    Alpha_grid, Beta_grid = np.meshgrid(Alpha, Beta)
    
    # Definimos las frecuencias base
    F1_fixed = F  # Primera frecuencia base
    F2_fixed = alpha_range[1] * F  # Segunda frecuencia base (según alpha)
    
    # Calculamos la disonancia para cada punto en la grilla
    D_surface = np.array([
        [Dis2(beta * F1_fixed, A, F2_fixed, A, alpha, method) for beta in Beta]
        for alpha in Alpha
    ])
    
    # Graficamos la superficie
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Alpha_grid, Beta_grid, D_surface, cmap='viridis')
    
    ax.set_xlabel('Alpha (F2 variable / F2 fija)')
    ax.set_ylabel('Beta (F1 variable / F2 fija)')
    ax.set_zlabel('Dissonance')
    ax.set_title(f'Surface of Dissonance ({method})')
    plt.show()

# Prueba con un tono puro
F_pure = np.array([440])  # Frecuencia base en Hz
A_pure = np.array([1.0])  # Amplitud normalizada

dissonance_surface(F_pure, A_pure, (1.0, 2.0), (1.0, 2.0))
