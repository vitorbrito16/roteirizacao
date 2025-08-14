from ortools.sat.python import cp_model
from dataclasses import dataclass, field
from typing import List

# Define os dados
@dataclass
class Projetista:
    id: int
    nome: str
    carga_diaria: int  # em horas

@dataclass
class Nota:
    id: int
    descricao: str
    tempo_estimado: int  # em horas
    prioridade: int  # quanto menor o número, maior a prioridade

# Exemplo de dados
projetistas = [
    Projetista(id=0, nome="Ana", carga_diaria=8),
    Projetista(id=1, nome="Carlos", carga_diaria=8),
]

notas = [
    Nota(id=0, descricao="Nota A", tempo_estimado=3, prioridade=1),
    Nota(id=1, descricao="Nota B", tempo_estimado=2, prioridade=2),
    Nota(id=2, descricao="Nota C", tempo_estimado=4, prioridade=1),
    Nota(id=3, descricao="Nota D", tempo_estimado=1, prioridade=3),
]

# Ordena as notas por prioridade (menor valor = maior prioridade)
notas.sort(key=lambda n: n.prioridade)

# Cria o modelo
model = cp_model.CpModel()

# Variáveis: nota atribuída ao projetista
nota_vars = {}
for nota in notas:
    for proj in projetistas:
        nota_vars[(nota.id, proj.id)] = model.NewBoolVar(f"nota{nota.id}_proj{proj.id}")

# Cada nota deve ser atribuída a exatamente um projetista
for nota in notas:
    model.Add(sum(nota_vars[(nota.id, proj.id)] for proj in projetistas) == 1)

# Respeita a carga diária de cada projetista
for proj in projetistas:
    model.Add(
        sum(nota_vars[(nota.id, proj.id)] * nota.tempo_estimado for nota in notas)
        <= proj.carga_diaria
    )

# Cria o solver
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Exibe os resultados
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for proj in projetistas:
        print(f"\nProjetista: {proj.nome}")
        total = 0
        for nota in notas:
            if solver.Value(nota_vars[(nota.id, proj.id)]):
                print(f"  - Nota {nota.descricao} (Prioridade {nota.prioridade}, {nota.tempo_estimado}h)")
                total += nota.tempo_estimado
        print(f"  Total atribuído: {total}h")
else:
    print("Não foi possível encontrar uma solução viável.")


