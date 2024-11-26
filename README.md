# parkify-AI

### Monitoramento e Gerenciamento de Vagas em Tempo Real:

- **Predição de Disponibilidade:** Analisar padrões para prever quais vagas estarão disponíveis em determinados horários, com base em histórico de ocupação.
<br>

- **Notificações em Tempo Real:** Enviar alertas para motoristas sobre a disponibilidade de vagas.
<br>

- **Detecção Anômala:** Identificar comportamentos incomuns, como ocupação longa de vagas ou sensores indicando ocupação falsa.
<br>

- **Predição de Ocupação Futura:** Utilizar séries temporais para prever quando o estacionamento estará cheio ou quando certa vaga estara vazia (de acordo com o tempo em que uma vaga costuma ficar ocupada).
<br>

- **Aprendizado Contínuo ou Online (se possível):** Melhorar o desempenho com dados acumulados para tornar a recomendação mais precisa ao longo do tempo.

## Estrutura dos Dados

### **Colunas Geradas**

<div align="center">


| Coluna             | Descrição                                                                 |
|---------------------|---------------------------------------------------------------------------|
| `Vaga`             | Identificação única da vaga (ex.: `A1`, `B5`).                           |
| `Status`           | Indica se a vaga está ocupada (`0`) ou livre (`1`).                      |
| `Distance (cm)`    | Distância medida pelo sensor:                                             |
|                    |  ≤ 40: Vaga ocupada                                                     |
|                    |  > 40: Vaga livre                                                       |
| `Luminosidade`     | Nível de luz captado pelo sensor (ex.: ambientes escuros têm valores altos). |
| `Data`             | Dia em que a vaga foi monitorada (formato `DD/MM/AAAA`).                 |
| `Início do Período`| Horário em que o status da vaga começou (formato `HH:MM:SS`).            |
| `Fim do Período`   | Horário em que o status da vaga terminou (formato `HH:MM:SS`).           |
| `Duração do Período`| Total de tempo em que a vaga ficou no mesmo estado (`hh:mm:ss`).        |




</div>

---

## Exemplo de Dados

### Vaga Ocupada (Status = 0)
| Vaga | Status | Distance | Luminosidade | Data       | Início do Período | Fim do Período | Duração do Período |
|------|--------|----------|--------------|------------|-------------------|----------------|--------------------|
| A1   | 0      | 34       | 711          | 16/08/2024 | 06:23:00         | 07:15:05       | 00:52:05          |

### Vaga Livre (Status = 1)
| Vaga | Status | Distance | Luminosidade | Data       | Início do Período | Fim do Período | Duração do Período |
|------|--------|----------|--------------|------------|-------------------|----------------|--------------------|
| A1   | 1      | 150      | 556          | 16/08/2024 | 07:15:05         | 08:31:00       | 01:15:55          |

---

## Algoritmo de Geração de Dados Sintéticos

O repositório contém um algoritmo em Python para gerar dados simulados de ocupação e disponibilidade de vagas. Este algoritmo é essencial para treinar e validar os modelos de IA que realizam as predições e notificações.

### Pontos-Chave do Algoritmo

1. **Geração de Períodos Ocupados**:
   - Cria períodos aleatórios em que uma vaga está ocupada, com `distance ≤ 40` e `luminosidade ≥ 600`.

2. **Cálculo de Períodos Livres**:
   - Identifica automaticamente os períodos entre ocupações em que a vaga está livre, com `distance > 40` e `luminosidade < 600`.

3. **Salvamento Dinâmico**:
   - Os dados são salvos em um arquivo Excel (`training_data.xlsx`), que pode ser reutilizado em etapas de análise ou treinamento de IA.

4. **Logs de Execução**:
   - O algoritmo registra logs detalhados para monitorar seu progresso e identificar possíveis problemas.

