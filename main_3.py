from ortools.sat.python import cp_model
from dataclasses import dataclass, field
from typing import List
import csv

# Definição das entidades
@dataclass
class Projetista:
    id: int
    nome: str
    carga_diaria: int
    habilidades: List[str]

@dataclass
class Nota:
    id: int
    descricao: str
    tempo_estimado: int
    prioridade: int
    habilidades_requeridas: List[str]

# Exemplo de dados
projetistas = [
    Projetista(0, "Ana", 8, ["eletrica", "hidraulica"]),
    Projetista(1, "Bruno", 8, ["mecanica"]),
    Projetista(2, "Carlos", 8, ["eletrica", "mecanica"]),
]

notas = [
    Nota(0, "Nota A", 3, 1, ["eletrica"]),
    Nota(2, "Nota C", 4, 1, ["hidraulica"]),
    Nota(1, "Nota B", 2, 2, ["mecanica"]),
    Nota(1, "Nota F", 2, 2, ["mecanica"]),
    Nota(4, "Nota E", 2, 2, ["hidraulica"]),
    Nota(3, "Nota D", 5, 3, ["eletrica", "mecanica"]),
]

# Ordenar notas por prioridade (menor número = maior prioridade)
notas.sort(key=lambda n: n.prioridade)

# Modelo
model = cp_model.CpModel()

# Variáveis de decisão: nota atribuída ao projetista
nota_para_projetista = {}
for nota in notas:
    for proj in projetistas:
        if all(hab in proj.habilidades for hab in nota.habilidades_requeridas):
            nota_para_projetista[(nota.id, proj.id)] = model.NewBoolVar(f"nota{nota.id}_proj{proj.id}")

# Restrição: cada nota atribuída a apenas um projetista elegível
for nota in notas:
    elegiveis = [nota_para_projetista[(nota.id, proj.id)] for proj in projetistas if (nota.id, proj.id) in nota_para_projetista]
    model.Add(sum(elegiveis) == 1)

# Restrição: carga horária por projetista
for proj in projetistas:
    carga_total = sum(nota.tempo_estimado * nota_para_projetista[(nota.id, proj.id)]
                      for nota in notas if (nota.id, proj.id) in nota_para_projetista)
    model.Add(carga_total <= proj.carga_diaria)

# Solução
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Exportar resultados para CSV
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    with open("distribuicao_notas.csv", mode="w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Nota", "Projetista", "Tempo Estimado", "Prioridade"])
        for nota in notas:
            for proj in projetistas:
                if (nota.id, proj.id) in nota_para_projetista and solver.Value(nota_para_projetista[(nota.id, proj.id)]) == 1:
                    writer.writerow([nota.descricao, proj.nome, nota.tempo_estimado, nota.prioridade])
    print("Distribuição exportada para 'distribuicao_notas.csv'.")
else:
    print("Não foi possível encontrar uma solução viável.")

