from ortools.sat.python import cp_model

# Dados de exemplo
notas = [
    {"id": 1, "descricao": "Nota A", "tempo": 2, "prioridade": 3},
    {"id": 2, "descricao": "Nota B", "tempo": 4, "prioridade": 2},
    {"id": 3, "descricao": "Nota C", "tempo": 3, "prioridade": 1},
    {"id": 4, "descricao": "Nota D", "tempo": 1, "prioridade": 2},
    {"id": 5, "descricao": "Nota E", "tempo": 2, "prioridade": 1},
]

projetistas = [
    {"id": 0, "nome": "Ana", "carga_diaria": 6},
    {"id": 1, "nome": "Bruno", "carga_diaria": 6},
]

# Inicializa o modelo
model = cp_model.CpModel()

# Variáveis: nota atribuída ao projetista
nota_vars = {}
for nota in notas:
    for proj in projetistas:
        nota_vars[(nota["id"], proj["id"])] = model.NewBoolVar(f"nota{nota['id']}_proj{proj['id']}")

# Restrição: cada nota deve ser atribuída a apenas um projetista
for nota in notas:
    model.Add(sum(nota_vars[(nota["id"], proj["id"])] for proj in projetistas) == 1)

# Restrição: carga diária por projetista
for proj in projetistas:
    model.Add(
        sum(nota_vars[(nota["id"], proj["id"])] * nota["tempo"] for nota in notas)
        <= proj["carga_diaria"]
    )

# Objetivo: balancear carga entre projetistas (minimizar diferença de tempo)
cargas = []
for proj in projetistas:
    carga = model.NewIntVar(0, proj["carga_diaria"], f"carga_proj{proj['id']}")
    model.Add(carga == sum(nota_vars[(nota["id"], proj["id"])] * nota["tempo"] for nota in notas))
    cargas.append(carga)

max_carga = model.NewIntVar(0, sum(nota["tempo"] for nota in notas), "max_carga")
min_carga = model.NewIntVar(0, sum(nota["tempo"] for nota in notas), "min_carga")
model.AddMaxEquality(max_carga, cargas)
model.AddMinEquality(min_carga, cargas)
model.Minimize(max_carga - min_carga)

# Resolve o modelo
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Exibe resultados
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for proj in projetistas:
        print(f"\nProjetista: {proj['nome']}")
        total = 0
        for nota in notas:
            if solver.Value(nota_vars[(nota["id"], proj["id"])]) == 1:
                print(f"  - {nota['descricao']} ({nota['tempo']}h)")
                total += nota["tempo"]
        print(f"  Total atribuído: {total}h")
else:
    print("Nenhuma solução encontrada.")


