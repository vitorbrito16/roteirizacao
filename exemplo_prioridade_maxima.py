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
    Cria um objetivo hierárquico:
    1. Maximiza a prioridade das tarefas (mais importante).
    2. Maximiza o esforco total das tarefas atribuídas (desempate).
    3. Balanceia a carga de trabalho (ajuste fino).
    """
    # --- Cálculo dos Scores para cada objetivo ---

    # 1. Score de Prioridade
    # Como prioridade '1' é melhor que '2', precisamos inverter o valor para maximização.
    # Uma tarefa de prioridade 1 valerá mais pontos que uma de prioridade 2.
    if not tarefas:
        max_prioridade = 1
    else:
        max_prioridade = max(t.prioridade for t in tarefas)
    
    score_total_prioridade = modelo.NewIntVar(0, len(tarefas) * max_prioridade, "score_prioridade")
    modelo.Add(score_total_prioridade == sum(
        (max_prioridade + 1 - tarefa.prioridade) * tarefa_recurso.get((tarefa.nota, recurso.matricula), 0)
        for tarefa in tarefas
        for recurso in recursos
    ))

    # 2. Score de esforco
    esforco_total_atribuido = modelo.NewIntVar(0, sum(t.esforco for t in tarefas), "esforco_total")
    modelo.Add(esforco_total_atribuido == sum(
        tarefa.esforco * tarefa_recurso.get((tarefa.nota, recurso.matricula), 0)
        for tarefa in tarefas
        for recurso in recursos
    ))

    # 3. Score de Balanceamento (Minimização da Diferença)
    variaveis_carga = []
    carga_maxima_geral = sum(t.esforco for t in tarefas)
    for recurso in recursos:
        carga_recurso = modelo.NewIntVar(0, carga_maxima_geral, f"carga_{recurso.matricula}")
        modelo.Add(
            carga_recurso == sum(
                tarefa.esforco * tarefa_recurso.get((tarefa.nota, recurso.matricula), 0)
                for tarefa in tarefas
            )
        )
        variaveis_carga.append(carga_recurso)
        
    carga_max = modelo.NewIntVar(0, carga_maxima_geral, "carga_max")
    carga_min = modelo.NewIntVar(0, carga_maxima_geral, "carga_min")
    modelo.AddMaxEquality(carga_max, variaveis_carga)
    modelo.AddMinEquality(carga_min, variaveis_carga)    
    diferenca_carga = modelo.NewIntVar(0, carga_maxima_geral, "diferenca_carga")
    modelo.Add(diferenca_carga == carga_max - carga_min)

    # --- Definição dos Pesos para a Hierarquia ---
    # O peso do nível superior deve ser maior que a soma máxima possível de todos os níveis inferiores.
    
    # O esforco total é um bom delimitador para o nível de balanceamento.
    PESO_BALANCEAMENTO = 1
    
    # O peso da prioridade deve ser maior que o esforco total máximo possível.
    # PESO_ESFORCO = carga_maxima_geral + 1
    PESO_ESFORCO = 3
    
    # O peso da prioridade deve ser maior que o score de esforco máximo possível
    # PESO_PRIORIDADE = (carga_maxima_geral + 1) * (len(tarefas) + 1)
    PESO_PRIORIDADE = 6

    # --- Combinação dos Objetivos ---
    modelo.Maximize(
        (score_total_prioridade * PESO_PRIORIDADE) +
        (esforco_total_atribuido * PESO_ESFORCO) -
        (diferenca_carga * PESO_BALANCEAMENTO)
    )

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
        df.to_csv("./output/distribuicao_tarefas_prioridade_maxima.csv", index=False, encoding="utf-8")

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
