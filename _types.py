from dataclasses import dataclass
from typing import List


@dataclass
class Recurso:
    matricula: str
    nome: str
    nucleo: str
    disponibilidade: int
    habilidades: List[str]


@dataclass
class Tarefa:
    nota: int
    grupo: str
    codigo: str
    esforco: float
    prioridade: int
    habilidades: List[str]
