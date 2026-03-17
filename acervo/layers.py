"""Capa de conocimiento — distingue conocimiento universal del personal.

Capa 1 (UNIVERSAL): hechos verificables del mundo (ciudades, países, hechos históricos).
Capa 2 (PERSONAL): conocimiento específico del usuario (proyectos, relaciones, preferencias).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


class Layer(Enum):
    UNIVERSAL = 1  # Capa 1: conocimiento del mundo, verificable externamente
    PERSONAL = 2   # Capa 2: conocimiento del usuario, dicho por él mismo


# Tipos de fuente de un nodo o arista
Source = Literal["world", "user_assertion"]

# Estados posibles de un nodo
NodeStatus = Literal["complete", "incomplete", "pending_verification"]
