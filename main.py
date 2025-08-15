import csv
import os
from typing import List

import pandas as pd
from dotenv import load_dotenv
from ortools.sat.python import cp_model

from _types import Recurso, Tarefa

load_dotenv()


def obter_recursos(caminho):
    recursos: List[Recurso] = []
    df = pd.read_csv(caminho, encoding="utf-8", sep=";")
    for _, row in df.iterrows():
        habilidades = row["habilidades"].split(",")
        recursos.append(
            Recurso(
                row["matricula"],
                row["nome"],
                row["nucleo"],
                int(row["disponibilidade"]),
                habilidades,
            )
        )
    return recursos, df


def obter_tarefas(caminho):
    tarefas: List[Tarefa] = []
    df = pd.read_csv(caminho, encoding="utf-8", sep=";")
    for _, row in df.iterrows():
        habilidades = row["habilidades"].split(",")
        tarefas.append(
            Tarefa(
                int(row["nota"]),
                row["grupo"],
                row["codigo"],
                int(row["custo"]),
                int(row["prioridade"]),
                habilidades,
            )
        )
    return tarefas, df


def aplicar_prioridade(tarefas: List[Tarefa]):
    # Ordenar notas por prioridade (menor número = maior prioridade)
    tarefas.sort(key=lambda n: n.prioridade)
    return tarefas


def aplicar_restricoes(modelo, tarefas, recursos):
    # Variáveis de decisão: nota atribuída ao projetista
    tarefa_recurso = {}
    for tarefa in tarefas:
        for recurso in recursos:
            if all(hab in recurso.habilidades for hab in tarefa.habilidades):
                tarefa_recurso[(tarefa.nota, recurso.matricula)] = modelo.NewBoolVar(
                    f"tarefa{tarefa.nota}_proj{recurso.matricula}"
                )

    # Restrição: cada tarefa atribuída a apenas um projetista elegível
    for tarefa in tarefas:
        elegiveis = [
            tarefa_recurso[(tarefa.nota, recurso.matricula)]
            for recurso in recursos
            if (tarefa.nota, recurso.matricula) in tarefa_recurso
        ]
        modelo.Add(sum(elegiveis) == 1)

    # Restrição: carga horária por projetista
    for recurso in recursos:
        carga_total = sum(
            tarefa.custo * tarefa_recurso[(tarefa.nota, recurso.matricula)]
            for tarefa in tarefas
            if (tarefa.nota, recurso.matricula) in tarefa_recurso
        )
        modelo.Add(carga_total <= recurso.disponibilidade)

    return modelo, tarefa_recurso


# def aplicar_objetivo(modelo, tarefas, recursos):
#     cargas = []
#     for recurso in recursos:
#         carga = modelo.NewIntVar(
#             0, recurso.disponibilidade, f"carga_recurso{recurso.matricula}"
#         )
#         modelo.Add(carga == sum(tarefa.custo for tarefa in tarefas))
#         cargas.append(carga)

#     max_carga = modelo.NewIntVar(
#         0, sum(tarefa.custo for tarefa in tarefas), "max_carga"
#     )
#     min_carga = modelo.NewIntVar(
#         0, sum(tarefa.custo for tarefa in tarefas), "min_carga"
#     )
#     modelo.AddMaxEquality(max_carga, cargas)
#     modelo.AddMinEquality(min_carga, cargas)
#     modelo.Minimize(max_carga - min_carga)

#     return modelo, cargas


def aplicar_objetivo_balanceamento(modelo, tarefas, recursos, tarefa_recurso):
    # Variáveis auxiliares para carga total por recurso
    carga_por_recurso = {}
    for recurso in recursos:
        carga = sum(
            tarefa.custo * tarefa_recurso[(tarefa.nota, recurso.matricula)]
            for tarefa in tarefas
            if (tarefa.nota, recurso.matricula) in tarefa_recurso
        )
        carga_por_recurso[recurso.matricula] = carga

    # Variável para carga máxima entre os recursos
    carga_max = modelo.NewIntVar(
        0, sum(tarefa.custo for tarefa in tarefas), "carga_max"
    )
    carga_min = modelo.NewIntVar(
        0, sum(tarefa.custo for tarefa in tarefas), "carga_min"
    )

    modelo.AddMaxEquality(carga_max, carga_por_recurso)
    modelo.AddMinEquality(carga_min, carga_por_recurso)
    modelo.Minimize(carga_max - carga_min)

    return modelo, carga_por_recurso


def solucionar_modelo(modelo):
    # Solução
    solver = cp_model.CpSolver()
    status = solver.Solve(modelo)
    return status, solver


def exportar_resultado(status, solver, tarefas, recursos, tarefa_recurso):
    # Exportar resultados para CSV
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        dados = []
        for tarefa in tarefas:
            for recurso in recursos:
                chave = (tarefa.nota, recurso.matricula)
                if chave in tarefa_recurso and solver.Value(tarefa_recurso[chave]) == 1:
                    dados.append(
                        {
                            "nota": tarefa.nota,
                            "matricula": recurso.matricula,
                            "nome": recurso.nome,
                            "custo": tarefa.custo,
                            "prioridade": tarefa.prioridade,
                        }
                    )

            df = pd.DataFrame(dados)
            df.to_csv("distribuicao_tarefas.csv", index=False, encoding="utf-8")

        print("Distribuição exportada para 'distribuicao_tarefas.csv'.")
    else:
        print(f"Não foi possível encontrar uma solução viável: {status}")


def main():
    CAMINHO_RECURSOS = os.getenv("CAMINHO_RECURSOS")
    CAMINHO_TAREFAS = os.getenv("CAMINHO_TAREFAS")

    tarefas, df_tarefas = obter_tarefas(CAMINHO_TAREFAS)
    recursos, df_recursos = obter_recursos(CAMINHO_RECURSOS)
    # print(df_tarefas.head())
    # print(df_recursos.head())

    tarefas_priorizadas = aplicar_prioridade(tarefas)
    modelo = cp_model.CpModel()

    modelo_restrito, tarefa_recurso = aplicar_restricoes(
        modelo, tarefas_priorizadas, recursos
    )
    modelo_final, carga_por_recurso = aplicar_objetivo_balanceamento(
        modelo_restrito, tarefas_priorizadas, recursos, tarefa_recurso
    )
    status, solver = solucionar_modelo(modelo_final)

    exportar_resultado(status, solver, tarefas_priorizadas, recursos, tarefa_recurso)


if __name__ == "__main__":
    main()
