import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.consonance_models import D2

def DisSurface(F1, A1, F2, A2, alpha_range, beta_range, method='Vassilakis'):
    """
    Calcula una superficie de disonancia variando tanto alpha como beta.
    
    Args:
        F1, A1: Frecuencias y amplitudes del primer sonido
        F2, A2: Frecuencias y amplitudes del segundo sonido
        alpha_range: Array de valores para alpha (ratio F2 variable / F1 fijo)
        beta_range: Array de valores para beta (ratio F1 variable / F2 fijo)
        method: 'Sethares1' o 'Vassilakis'
        
    Returns:
        Alpha, Beta, D: Meshgrids para alpha, beta y los valores de disonancia
    """
    Alpha, Beta = np.meshgrid(alpha_range, beta_range)
    D = np.zeros_like(Alpha)
    
    for i in range(Alpha.shape[0]):
        for j in range(Alpha.shape[1]):
            # Primero escalamos F1 por beta
            scaled_F1 = F1 * Beta[i,j]
            # Luego escalamos F2 por alpha
            scaled_F2 = F2 * Alpha[i,j]
            # Calculamos la disonancia entre los dos sonidos escalados
            D[i,j] = D2(scaled_F1, A1, scaled_F2, A2, method)
    
    return Alpha, Beta, D

def plot_dissonance_surface(Alpha, Beta, D):
    """Visualiza la superficie de disonancia en 3D."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(Alpha, Beta, D, cmap='viridis', 
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('Alpha (F2 variable / F1 fijo)')
    ax.set_ylabel('Beta (F1 variable / F2 fijo)')
    ax.set_zlabel('Disonancia')
    ax.set_title('Superficie de Disonancia')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# Ejemplo con tonos puros

# Definimos un tono puro (frecuencia fundamental + armónicos débiles)
F_pure = np.array([440.0])  # Frecuencia fundamental
A_pure = np.array([1.0])    # Amplitud
    
# Rangos para alpha y beta
alpha_vals = np.linspace(1, 2.0, 50)  # De media a doble frecuencia
beta_vals = np.linspace(2, 2.0, 50)

# Calculamos la superficie
Alpha, Beta, D = DisSurface(F_pure, A_pure, F_pure, A_pure, 
                            alpha_vals, beta_vals, method='Vassilakis')

# Visualizamos
plot_dissonance_surface(Alpha, Beta, D)
