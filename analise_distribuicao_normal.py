"""
Trabalho: Distribuição Normal (Curva de Gauss)
Análise da coluna "math score" da base Students Performance in Exams.

Bibliotecas permitidas: csv, math e matplotlib (somente para os gráficos).
Todos os cálculos estatísticos são feitos manualmente.
"""

import csv
import math
import os

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
PASTA = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_CSV = os.path.join(PASTA, "StudentsPerformance.csv")
ARQUIVO_GRAFICO = os.path.join(PASTA, "histograma_curva_normal.png")
ARQUIVO_BOXPLOT = os.path.join(PASTA, "boxplot.png")
COLUNA = "math score"
NOTAS_Z = [50, 75, 90]


def linha(titulo=""):
    print()
    print("=" * 60)
    if titulo:
        print(titulo)
        print("=" * 60)


# ----------------------------------------------------------------------
# Parte B — Leitura e organização dos dados
# ----------------------------------------------------------------------
def carregar_notas(caminho, coluna):
    """Lê o CSV com csv.DictReader e devolve a lista de notas válidas
    e a lista de problemas encontrados (linha, valor)."""
    notas = []
    problemas = []

    with open(caminho, newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)

        if coluna not in leitor.fieldnames:
            raise ValueError(f"A coluna '{coluna}' não existe no arquivo.")

        # a linha 1 é o cabeçalho, então os dados começam na linha 2
        for numero_linha, registro in enumerate(leitor, start=2):
            valor = registro.get(coluna)

            if valor is None or valor.strip() == "":
                problemas.append((numero_linha, "valor ausente"))
                continue

            try:
                nota = int(valor.strip())
            except ValueError:
                problemas.append((numero_linha, f"valor inválido: {valor!r}"))
                continue

            if nota < 0 or nota > 100:
                problemas.append((numero_linha, f"fora do intervalo 0-100: {nota}"))
                continue

            notas.append(nota)

    return notas, problemas


# ----------------------------------------------------------------------
# Parte C — Medidas estatísticas (sem bibliotecas estatísticas)
# ----------------------------------------------------------------------
def ordenar(dados):
    return sorted(dados)


def soma(dados):
    total = 0
    for x in dados:
        total += x
    return total


def media(dados):
    return soma(dados) / len(dados)


def mediana(dados):
    ordenados = ordenar(dados)
    n = len(ordenados)
    meio = n // 2
    if n % 2 == 1:
        return ordenados[meio]
    return (ordenados[meio - 1] + ordenados[meio]) / 2


def modas(dados):
    frequencias = {}
    for x in dados:
        frequencias[x] = frequencias.get(x, 0) + 1

    maior_freq = max(frequencias.values())
    lista_modas = [valor for valor, freq in frequencias.items() if freq == maior_freq]
    return sorted(lista_modas), maior_freq


def variancia_amostral(dados, m):
    n = len(dados)
    soma_quadrados = 0
    for x in dados:
        soma_quadrados += (x - m) ** 2
    return soma_quadrados / (n - 1)


def coeficiente_assimetria(dados, m, s):
    """Coeficiente de assimetria amostral (Fisher-Pearson ajustado):
    g = n / ((n-1)(n-2)) * soma(((x - média) / s) ** 3)"""
    n = len(dados)
    soma_cubos = 0
    for x in dados:
        soma_cubos += ((x - m) / s) ** 3
    return (n / ((n - 1) * (n - 2))) * soma_cubos


def classificar_assimetria(g):
    if abs(g) < 0.1:
        return "aproximadamente simétrica"
    lado = "negativa (cauda à esquerda)" if g < 0 else "positiva (cauda à direita)"
    if abs(g) < 0.5:
        return "leve assimetria " + lado
    if abs(g) < 1:
        return "assimetria moderada " + lado
    return "assimetria forte " + lado


# ----------------------------------------------------------------------
# Parte D e E — Regra empírica e escore z
# ----------------------------------------------------------------------
def regra_empirica(dados, m, s):
    resultados = []
    esperados = {1: 68.0, 2: 95.0, 3: 99.7}
    n = len(dados)
    for k in (1, 2, 3):
        limite_inf = m - k * s
        limite_sup = m + k * s
        quantidade = 0
        for x in dados:
            if limite_inf <= x <= limite_sup:
                quantidade += 1
        percentual = quantidade / n * 100
        resultados.append((k, limite_inf, limite_sup, quantidade, percentual, esperados[k]))
    return resultados


def escore_z(x, m, s):
    return (x - m) / s


def densidade_normal(x, m, s):
    return (1 / (s * math.sqrt(2 * math.pi))) * math.exp(-((x - m) ** 2) / (2 * s ** 2))


# ----------------------------------------------------------------------
# Parte F — Gráficos
# ----------------------------------------------------------------------
def gerar_histograma(dados, m, s, caminho):
    fig, ax = plt.subplots(figsize=(10, 6))

    # classes de largura 5, de 0 a 100
    classes = list(range(0, 105, 5))
    ax.hist(dados, bins=classes, density=True, color="#9ecae1",
            edgecolor="white", label="Notas de matemática")

    # curva normal teórica calculada com a fórmula da densidade
    xs = [i / 10 for i in range(0, 1001)]
    ys = [densidade_normal(x, m, s) for x in xs]
    ax.plot(xs, ys, color="#08519c", linewidth=2.5,
            label=f"Curva normal teórica (μ={m:.2f}, σ={s:.2f})")

    ax.axvline(m, color="#d62728", linewidth=2, label=f"Média = {m:.2f}")
    ax.axvline(m - s, color="#ff7f0e", linewidth=1.8, linestyle="--",
               label=f"μ − σ = {m - s:.2f}")
    ax.axvline(m + s, color="#ff7f0e", linewidth=1.8, linestyle="--",
               label=f"μ + σ = {m + s:.2f}")

    ax.set_title("Distribuição das notas de matemática com curva normal teórica")
    ax.set_xlabel("Nota de matemática")
    ax.set_ylabel("Densidade")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def gerar_boxplot(dados, caminho):
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.boxplot(dados, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#9ecae1"),
               medianprops=dict(color="#d62728", linewidth=2))
    ax.set_title("Gráfico de caixa das notas de matemática")
    ax.set_ylabel("Nota de matemática")
    ax.set_xticks([])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
def main():
    # Parte B
    linha("PARTE B — LEITURA E ORGANIZAÇÃO DOS DADOS")
    notas, problemas = carregar_notas(ARQUIVO_CSV, COLUNA)
    total_lido = len(notas) + len(problemas)
    print(f"Arquivo: {os.path.basename(ARQUIVO_CSV)}")
    print(f"Coluna analisada: {COLUNA}")
    print(f"Registros lidos: {total_lido}")
    print(f"Notas válidas carregadas: {len(notas)}")
    if problemas:
        print(f"Valores ausentes ou inválidos: {len(problemas)}")
        for numero_linha, motivo in problemas:
            print(f"  linha {numero_linha}: {motivo}")
    else:
        print("Valores ausentes ou inválidos: nenhum")

    if len(notas) < 3:
        print("Dados insuficientes para a análise.")
        return

    # Parte C
    linha("PARTE C — MEDIDAS ESTATÍSTICAS")
    n = len(notas)
    ordenadas = ordenar(notas)
    menor = ordenadas[0]
    maior = ordenadas[-1]
    amplitude = maior - menor
    m = media(notas)
    md = mediana(notas)
    lista_modas, freq_moda = modas(notas)
    var = variancia_amostral(notas, m)
    s = math.sqrt(var)
    g = coeficiente_assimetria(notas, m, s)

    print(f"Quantidade de dados:      {n}")
    print(f"Menor nota:               {menor}")
    print(f"Maior nota:               {maior}")
    print(f"Amplitude total:          {amplitude}")
    print(f"Média aritmética:         {m:.4f}")
    print(f"Mediana:                  {md}")
    texto_modas = ", ".join(str(x) for x in lista_modas)
    rotulo = "Moda" if len(lista_modas) == 1 else "Modas"
    print(f"{rotulo + ':':<26}{texto_modas} (frequência {freq_moda})")
    print(f"Variância amostral:       {var:.4f}")
    print(f"Desvio-padrão amostral:   {s:.4f}")
    print(f"Coeficiente de assimetria: {g:.4f} -> {classificar_assimetria(g)}")

    # Parte D
    linha("PARTE D — REGRA 68–95–99,7")
    print(f"{'Intervalo':<10}{'Limites':<22}{'Qtd':>6}{'Obs. (%)':>11}{'Esp. (%)':>11}")
    resultados_regra = regra_empirica(notas, m, s)
    for k, inf, sup, qtd, perc, esp in resultados_regra:
        limites = f"[{inf:.2f}; {sup:.2f}]"
        print(f"{'μ ± ' + str(k) + 'σ':<10}{limites:<22}{qtd:>6}{perc:>10.2f}%{esp:>10.1f}%")

    # Parte E
    linha("PARTE E — ESCORES PADRONIZADOS (z)")
    casos = [("Menor nota", menor), ("Maior nota", maior)]
    casos += [(f"Nota {x}", x) for x in NOTAS_Z]
    for descricao, x in casos:
        z = escore_z(x, m, s)
        situacao = "possível valor atípico" if abs(z) > 3 else "dentro do esperado"
        print(f"{descricao:<12} x = {x:>3}  z = {z:>7.4f}  ({situacao})")

    atipicos = [x for x in notas if abs(escore_z(x, m, s)) > 3]
    print()
    if atipicos:
        print(f"Notas com |z| > 3 na base: {len(atipicos)} -> {sorted(atipicos)}")
    else:
        print("Nenhuma nota da base apresenta |z| > 3.")

    # Parte F
    linha("PARTE F — REPRESENTAÇÃO GRÁFICA")
    gerar_histograma(notas, m, s, ARQUIVO_GRAFICO)
    print(f"Histograma salvo em: {os.path.basename(ARQUIVO_GRAFICO)}")
    gerar_boxplot(notas, ARQUIVO_BOXPLOT)
    print(f"Gráfico de caixa salvo em: {os.path.basename(ARQUIVO_BOXPLOT)}")

    # Resumo para apoiar a Parte G
    linha("RESUMO PARA A CONCLUSÃO (PARTE G)")
    print(f"Diferença entre média e mediana: {abs(m - md):.4f}")
    for k, _, _, _, perc, esp in resultados_regra:
        print(f"μ ± {k}σ: observado {perc:.2f}% x esperado {esp}% "
              f"(diferença {perc - esp:+.2f} p.p.)")
    print(f"Assimetria: {g:.4f} -> {classificar_assimetria(g)}")
    print(f"Valores atípicos (|z| > 3): {len(atipicos)}")


if __name__ == "__main__":
    main()
