"""
Configuración de pytest para HOC.
Añade el directorio raíz al path para importar el paquete.
"""
import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
