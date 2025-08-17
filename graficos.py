from matplotlib import pyplot as plt
import numpy as np

def prioridade_x_esforco(peso_prioridade=1, peso_esforco=1, esforco_maximo_geral=2000):
  # Solução A: Score de prioridade = 500, esforco máximo = 2000
  sol_A = {'prioridade': 6, 'esforco': 4}

  # Solução B: Score de prioridade um pouco maior = 501, esforco zero = 0
  sol_B = {'prioridade': 7, 'esforco': 3}

  # Calcular o valor final do objetivo (ignorando o balanceamento por enquanto)
  valor_A = (sol_A['prioridade'] * peso_prioridade) + (sol_A['esforco'] * peso_esforco)
  valor_B = (sol_B['prioridade'] * peso_prioridade) + (sol_B['esforco'] * peso_esforco)

  # --- Plotando o Gráfico 1 ---
  plt.figure(figsize=(10, 7))
  plt.plot(sol_A['esforco'], sol_A['prioridade'], 'ro', markersize=15, label=f'Solução A (Valor Obj: {valor_A:,.0f})')
  plt.plot(sol_B['esforco'], sol_B['prioridade'], 'go', markersize=15, label=f'Solução B (Valor Obj: {valor_B:,.0f})')

  # Seta indicando a escolha do solver
  plt.annotate('Escolha do Solver', xy=(sol_B['esforco'], sol_B['prioridade']),
              xytext=(sol_A['esforco'] * 0.6, sol_A['prioridade'] * 1.001),
              arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
              fontsize=12, ha='center')

  plt.title('Gráfico 1: Dominância da Prioridade sobre o esforco', fontsize=16)
  plt.xlabel('esforco Total Atribuído', fontsize=12)
  plt.ylabel('Score Total de Prioridade', fontsize=12)
  plt.xlim(-100, esforco_maximo_geral + 100)
  plt.legend()
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.show()

  print(f"A Solução B é melhor? {valor_B > valor_A}")

def esforco_x_balanceamento(peso_esforco=1, peso_balanceamento=1, esforco_maximo_geral=2000):
  sol_A = {'esforco': 3, 'balanceamento': 1}

  sol_B = {'esforco': 4, 'balanceamento': 1}

  # Calcular o valor final do objetivo (ignorando a prioridade)
  valor_A = (sol_A['esforco'] * peso_esforco) - (sol_A['balanceamento'] * peso_balanceamento)
  valor_B = (sol_B['esforco'] * peso_esforco) - (sol_B['balanceamento'] * peso_balanceamento)

  # --- Plotando o Gráfico 2 ---
  plt.figure(figsize=(10, 7))
  plt.plot(sol_A['balanceamento'], sol_A['esforco'], 'ro', markersize=15, label=f'Solução A (Valor Obj: {valor_A:,.0f})')
  plt.plot(sol_B['balanceamento'], sol_B['esforco'], 'go', markersize=15, label=f'Solução B (Valor Obj: {valor_B:,.0f})')

  # Seta indicando a escolha do solver
  plt.annotate('Escolha do Solver', xy=(sol_B['balanceamento'], sol_B['esforco']),
              xytext=(sol_A['balanceamento'] + 250, sol_A['esforco'] + 0.5),
              arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
              fontsize=12, ha='center')

  plt.title('Gráfico 2: Dominância do Esforco sobre o Balanceamento', fontsize=16)
  plt.xlabel('Diferença de Carga (Menor é Melhor)', fontsize=12)
  plt.ylabel('Esforco Total Atribuído', fontsize=12)
  plt.legend()
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.show()

  print(f"A Solução B é melhor? {valor_B > valor_A}")

def representacao_grafica(peso_prioridade=1, peso_esforco=1, peso_balanceamento=1):
  # Suponha uma solução ótima encontrada
  score_prioridade_final = 550
  esforco_final = 1800
  balanceamento_final = 150

  # Calculando a contribuição ponderada de cada um
  contrib_prioridade = score_prioridade_final * peso_prioridade
  contrib_esforco = esforco_final * peso_esforco
  contrib_balanceamento = balanceamento_final * peso_balanceamento # Usamos valor absoluto para o gráfico

  componentes = ['Prioridade', 'esforco', 'Balanceamento']
  valores = [contrib_prioridade, contrib_esforco, contrib_balanceamento]

  # --- Plotando o Gráfico 3 ---
  plt.figure(figsize=(10, 7))
  bars = plt.bar(componentes, valores, color=['#003f5c', '#7a5195', '#ef5675'])

  # Usar escala logarítmica para tornar as barras visíveis
  plt.yscale('log')

  plt.title('Gráfico 3: Contribuição de Cada Componente (Escala Logarítmica)', fontsize=16)
  plt.ylabel('Valor Ponderado (Log)', fontsize=12)
  plt.grid(True, which="both", ls="--", alpha=0.6)

  # Adicionar rótulos de dados
  for bar in bars:
      yval = bar.get_height()
      plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:,.0f}', va='bottom', ha='center', fontsize=10) # va: vertical alignment

  plt.show()

def campo_de_decisao():
  # --- 1. Nossos novos pesos ---
  pesos_percentuais = {'prioridade': 60, 'esforco': 30}

  # --- 2. Preparação da Grade (agora em escala normalizada 0-1000) ---
  esforco_norm_vals = np.linspace(0, 1000, 100)
  prioridade_norm_vals = np.linspace(0, 1000, 100)
  E, P, = np.meshgrid(esforco_norm_vals, prioridade_norm_vals)

  # Calculamos o valor do objetivo PONDERADO para cada ponto
  Z_objetivo = (P * pesos_percentuais['prioridade']) + (E * pesos_percentuais['esforco'])

  # --- 3. Plotando o Gráfico de Contorno ---
  plt.figure(figsize=(12, 8))
  contour_fill = plt.contourf(E, P, Z_objetivo, levels=20, cmap='cividis', alpha=0.8)
  cbar = plt.colorbar(contour_fill)
  cbar.set_label('Valor da Função Objetivo Ponderada', fontsize=12)

  # --- 4. A Seta de Otimização (agora na diagonal!) ---
  # A direção reflete a proporção dos pesos (30 para esforço, 60 para prioridade)
  plt.arrow(500, 400, 300 * 0.5, 600 * 0.5, head_width=20, head_length=30, fc='magenta', ec='magenta', width=5)
  plt.text(500, 350, 'Direção de\nOtimização (Trade-off)', color='magenta', ha='center', fontsize=12, weight='bold')

  plt.title('Gráfico: Campo de Decisão Ponderado (Prioridade vs. Esforço)', fontsize=16)
  plt.xlabel('Esforço Total (Normalizado 0-1000)', fontsize=12)
  plt.ylabel('Score de Prioridade (Normalizado 0-1000)', fontsize=12)
  plt.grid(True, linestyle='--', alpha=0.3)
  plt.axis('equal') # Garante que a proporção visual esteja correta
  plt.show()

def main():
  # prioridade_x_esforco()
  # esforco_x_balanceamento()
  # representacao_grafica()
  campo_de_decisao()

if __name__ == "__main__":
  main()