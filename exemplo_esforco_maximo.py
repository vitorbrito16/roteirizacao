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
                int(row["esforco"]),
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

    # Restrição: cada tarefa atribuída a apenas um recurso elegível
    for tarefa in tarefas:
        elegiveis = [
            tarefa_recurso[(tarefa.nota, recurso.matricula)]
            for recurso in recursos
            if (tarefa.nota, recurso.matricula) in tarefa_recurso
        ]
        modelo.Add(sum(elegiveis) <= 1)

    # Restrição: carga horária por projetista
    for recurso in recursos:
        carga_total = sum(
            tarefa.esforco * tarefa_recurso[(tarefa.nota, recurso.matricula)]
            for tarefa in tarefas
            if (tarefa.nota, recurso.matricula) in tarefa_recurso
        )
        modelo.Add(carga_total <= recurso.disponibilidade)

    return modelo, tarefa_recurso

def aplicar_objetivos(modelo, tarefas, recursos, tarefa_recurso):
    """
    Cria um objetivo com duas metas:
    1. Maximizar o esforco total das tarefas atribuídas (maior prioridade).
    2. Balancear a carga de trabalho entre os recursos (menor prioridade).
    """
    # --- Parte 1: Maximizar o trabalho feito ---
    esforco_total_atribuido = sum(
        tarefa.esforco * tarefa_recurso.get((tarefa.nota, recurso.matricula), 0)
        for tarefa in tarefas
        for recurso in recursos
    )
    
    # --- Parte 2: Balancear a carga ---
    # Esta parte é idêntica à sua função de balanceamento
    variaveis_carga = []
    for recurso in recursos:
        carga_recurso = modelo.NewIntVar(0, recurso.disponibilidade, f"carga_{recurso.matricula}")
        modelo.Add(
            carga_recurso == sum(
                tarefa.esforco * tarefa_recurso.get((tarefa.nota, recurso.matricula), 0)
                for tarefa in tarefas
            )
        )
        variaveis_carga.append(carga_recurso)
        
    carga_max = modelo.NewIntVar(0, sum(tarefa.esforco for tarefa in tarefas), "carga_max")
    carga_min = modelo.NewIntVar(0, sum(tarefa.esforco for tarefa in tarefas), "carga_min")
    modelo.AddMaxEquality(carga_max, variaveis_carga)
    modelo.AddMinEquality(carga_min, variaveis_carga)
    
    diferenca_carga = carga_max - carga_min
    
    # --- Combinando os Objetivos ---
    # Damos um peso muito maior para o trabalho feito.
    # Por exemplo, cada ponto de 'esforco' vale 1000 pontos de "não-balanceamento".
    # Isso garante que o solver NUNCA sacrificará uma tarefa em prol de um melhor balanceamento.
    
    # Usamos .get() com valor padrão 0 para o caso de o par (tarefa, recurso) não ser válido (falta de habilidade).
    
    PESO_MAXIMIZACAO = 1000 
    
    modelo.Maximize( (esforco_total_atribuido * PESO_MAXIMIZACAO) - diferenca_carga )

    return modelo


def solucionar_modelo(modelo, tempo_limite=30.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tempo_limite
    status = solver.Solve(modelo)
    return status, solver


def exportar_resultado(status, solver, tarefas, recursos, tarefa_recurso):

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Status da solução: {solver.StatusName(status)} ({status})")
        print(f"Valor do objetivo alcançado: {solver.ObjectiveValue()}")
        if status == cp_model.OPTIMAL:
            print("Solução ótima encontrada.")
        if status == cp_model.FEASIBLE:
            print("Solução viável encontrada, mas não ótima.")

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
                            "esforco": tarefa.esforco,
                            "prioridade": tarefa.prioridade,
                        }
                    )
        df = pd.DataFrame(dados)
        df.to_csv("./output/distribuicao_tarefas_esforco_maximo.csv", index=False, encoding="utf-8")

        print("Distribuição exportada para 'distribuicao_tarefas.csv'.")
    else:
        print(f"Não foi possível encontrar uma solução viável: {status}")


def main():
    CAMINHO_RECURSOS = os.getenv("CAMINHO_RECURSOS")
    CAMINHO_TAREFAS = os.getenv("CAMINHO_TAREFAS")
    TEMPO_LIMITE = float(os.getenv("TEMPO_LIMITE", 30.0))

    tarefas, df_tarefas = obter_tarefas(CAMINHO_TAREFAS)
    recursos, df_recursos = obter_recursos(CAMINHO_RECURSOS)
    # print(df_tarefas.head())
    # print(df_recursos.head())

    tarefas_priorizadas = aplicar_prioridade(tarefas)
    modelo = cp_model.CpModel()

    modelo_restrito, tarefa_recurso = aplicar_restricoes(
        modelo, tarefas_priorizadas, recursos
    )
    modelo_final = aplicar_objetivos(
        modelo_restrito, tarefas_priorizadas, recursos, tarefa_recurso
    )
    status, solver = solucionar_modelo(modelo_final, TEMPO_LIMITE)

    exportar_resultado(status, solver, tarefas_priorizadas, recursos, tarefa_recurso)


if __name__ == "__main__":
    main()
