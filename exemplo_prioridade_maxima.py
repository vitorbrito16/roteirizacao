import os
from typing import List

import pandas as pd
from dotenv import load_dotenv
from ortools.sat.python import cp_model

from _types import Recurso, Tarefa

load_dotenv()


def obter_recursos(caminho):
    """
    Lê os dados dos recursos de um arquivo CSV.

    Args:
        caminho (str): O caminho para o arquivo CSV de recursos.

    Returns:
        tuple: Uma tupla contendo uma lista de objetos Recurso e o DataFrame original.
    """
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
    """
    Lê os dados das tarefas de um arquivo CSV.

    Args:
        caminho (str): O caminho para o arquivo CSV de tarefas.

    Returns:
        tuple: Uma tupla contendo uma lista de objetos Tarefa e o DataFrame original.
    """
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
    """
    Ordena a lista de tarefas com base na prioridade.
    A ordenação é feita in-place.

    Args:
        tarefas (List[Tarefa]): A lista de tarefas a ser ordenada.

    Returns:
        List[Tarefa]: A lista de tarefas ordenada por prioridade.
    """
    # Ordenar notas por prioridade (menor número = maior prioridade)
    tarefas.sort(key=lambda n: n.prioridade)
    return tarefas

def aplicar_restricoes(modelo, tarefas, recursos):
    """
    Aplica as restrições do problema ao modelo CP-SAT.

    Args:
        modelo (cp_model.CpModel): O objeto do modelo.
        tarefas (List[Tarefa]): A lista de tarefas.
        recursos (List[Recurso]): A lista de recursos.

    Returns:
        tuple: Uma tupla contendo o modelo com as restrições e o dicionário de variáveis de decisão.
    """
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
    Define a função objetivo para o modelo de otimização.
    O objetivo é maximizar uma combinação ponderada da prioridade e do esforço das tarefas.
    1. Maximiza a prioridade das tarefas (mais importante).
    2. Maximiza a utilizacao dos recursos.

    Args:
        modelo (cp_model.CpModel): O objeto do modelo.
        tarefas (List[Tarefa]): A lista de tarefas.
        recursos (List[Recurso]): A lista de recursos.
        tarefa_recurso (dict): Dicionário com as variáveis de decisão.

    Returns:
        cp_model.CpModel: O modelo com a função objetivo definida.
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

    # --- Definição dos Pesos para a Hierarquia ---
    # O peso do nível superior deve ser maior que a soma máxima possível de todos os níveis inferiores.
    
    modelo.Maximize(
        (score_total_prioridade * PESOS_PERCENTUAIS['prioridade']) +
        (esforco_total_atribuido * PESOS_PERCENTUAIS['esforco'])
    )

    return modelo

def solucionar_modelo(modelo, tempo_limite=30.0):
    """
    Resolve o modelo CP-SAT usando o solver.

    Args:
        modelo (cp_model.CpModel): O modelo a ser resolvido.
        tempo_limite (float, optional): O tempo máximo em segundos para o solver. Defaults to 30.0.

    Returns:
        tuple: Uma tupla contendo o status da solução e o objeto solver.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tempo_limite
    status = solver.Solve(modelo)
    return status, solver

def exportar_resultado(status, solver, tarefas, recursos, tarefa_recurso):
    """
    Exporta o resultado da otimização para um arquivo CSV se uma solução for encontrada.

    Args:
        status: O status da solução retornado pelo solver.
        solver: O objeto solver após a execução.
        tarefas (List[Tarefa]): A lista de tarefas.
        recursos (List[Recurso]): A lista de recursos.
        tarefa_recurso (dict): Dicionário com as variáveis de decisão.
    """
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
        df.to_csv("./data/distribuicao_tarefas_prioridade_maxima.csv", index=False, encoding="utf-8", sep=";")

        print("Distribuição exportada para 'distribuicao_tarefas.csv'.")
    else:
        print(f"Não foi possível encontrar uma solução viável: {status}")

def main():
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
    CAMINHO_RECURSOS = os.getenv("CAMINHO_RECURSOS")
    CAMINHO_TAREFAS = os.getenv("CAMINHO_TAREFAS")
    TEMPO_LIMITE = float(os.getenv("TEMPO_LIMITE", 30.0))
    PESOS_PERCENTUAIS = {
        "prioridade": int(os.getenv("PESO_PRIORIDADE", 60)),
        "esforco": int(os.getenv("PESO_ESFORCO", 40)),
    }
    print(f"Pesos Percentuais: {PESOS_PERCENTUAIS}")
    ESCALA_NORMALIZACAO = int(os.getenv("ESCALA_NORMALIZACAO", 1000))
    main()
