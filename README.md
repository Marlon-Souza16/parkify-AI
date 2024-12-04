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


<div align="center">


| Coluna              | Descrição                                                                 |
|----------------------|---------------------------------------------------------------------------|
| `Vaga`              | Identificação única da vaga (ex.: `A1`, `B5`).                           |
| `Status`            | Indica se a vaga está ocupada (`0`) ou livre (`1`).                      |
| `Data`              | Dia em que a vaga foi monitorada (formato `DD/MM/AAAA`).                 |
| `Day of Week`       | Dia da semana (0 = Segunda-feira, 6 = Domingo).                          |
| `Time (min)`        | Tempo em minutos desde o início do dia.                                  |
| `Time Period`       | Período do dia (`Early Morning`, `Morning`, `Afternoon`, `Evening`).     |
| `Is Weekend`        | Indica se é fim de semana (`0` = Não, `1` = Sim).                        |
| `Weekday_Hour`      | Interação entre dia da semana e horário (captura padrões semanais).      |

---

## Exemplo de Dados Processados

| Vaga | Status | Data       | Day of Week | Time (min) | Time Period | Is Weekend | Weekday_Hour |
|------|--------|------------|-------------|------------|-------------|------------|--------------|
| A1   | 0      | 16/08/2024 | 4           | 387        | Morning     | 0          | 1548         |
| A1   | 1      | 16/08/2024 | 4           | 720        | Afternoon   | 0          | 2880         |



</div>

---

## Algoritmo de Geração de Dados Sintéticos

O repositório contém um algoritmo em Python para gerar dados simulados de ocupação e disponibilidade de vagas. Este algoritmo é essencial para treinar e validar os modelos de IA que realizam as predições e notificações.
(Melhorar detalhamento)

---

## Algoritmo de IA

### **Modelo Utilizado**

O projeto utiliza o algoritmo **XGBoost**, configurado para realizar classificações rápidas e precisas, mesmo com grandes volumes de dados. O modelo é otimizado usando validação cruzada e ajuste de hiperparâmetros com `RandomizedSearchCV`.

### **Principais Recursos**
1. **Treinamento Paralelo:**
   - O treinamento utiliza todos os núcleos do processador para lidar com grandes volumes de dados.
2. **Otimização Automática:**
   - Os hiperparâmetros, como profundidade da árvore e taxa de aprendizado, são ajustados automaticamente para maximizar a acurácia.
3. **Predição em Tempo Real:**
   - Após treinado, o modelo pode ser usado para prever a disponibilidade de vagas em segundos.
4. **Salvamento do Modelo:**
   - O modelo é salvo em formato `.pkl`, permitindo reutilização futura sem necessidade de re-treinamento.

---

## Fluxo de Trabalho

1. **Pré-processamento dos Dados**:
   - Expansão granular para registros minuto a minuto.
   - Enriquecimento com variáveis derivadas (ex.: períodos do dia, finais de semana).

2. **Treinamento do Modelo**:
   - Utilização de XGBoost com paralelização total (`n_jobs=-1`).
   - Ajuste de hiperparâmetros com `RandomizedSearchCV`.

3. **Avaliação**:
   - Relatórios detalhados de acurácia, precisão, recall e F1-score.

4. **Reutilização**:
   - Salvamento do modelo treinado para uso em predições futuras.

---

## Detalhe sobre `Weekday_Hour`

A variável **`Weekday_Hour`** é uma *feature* criada para capturar a interação entre o **dia da semana** e o **horário** do dia. Essa interação ajuda o modelo a identificar padrões específicos que dependem tanto do horário quanto do dia da semana.

### **Como é Calculada**
- **`Day of Week`**: Representa o dia da semana (0 = Segunda-feira, 6 = Domingo).
- **`Time (min)`**: Representa o horário em minutos desde o início do dia (ex.: 12:00 = 720 minutos).

### **Por que ela é útil?**

#### **1. Padrões Semanais**
Certos horários podem ser mais movimentados em dias específicos. Por exemplo:
- Segunda-feira de manhã (horário de pico de trabalho).
- Sábado à tarde (mais visitantes em shoppings).

A multiplicação `Day of Week * Time (min)` cria uma *feature* que diferencia essas situações.

#### **2. Ajuda na Identificação de Combinações**
Diferencia, por exemplo:
- Segunda-feira às 10:00 (600 minutos * 0 = 0).
- Domingo às 10:00 (600 minutos * 6 = 3600).

O modelo consegue aprender que o mesmo horário tem padrões diferentes dependendo do dia.

#### **3. Evita Redundância**
Ao invés de criar variáveis separadas para **hora** e **dia**, a interação combina as informações em uma única *feature*.

### **Exemplo**

| Day of Week | Time (min) | Weekday_Hour |
|-------------|------------|--------------|
| 0 (Segunda) | 600        | 0            |
| 1 (Terça)   | 600        | 600          |
| 6 (Domingo) | 600        | 3600         |

Tendo isso em vista, segue um exemplo prático:
- Segunda-feira às 10:00 (600 minutos) é representada como `0` (início da semana).
- Domingo às 10:00 é representado como `3600`, destacando um horário e dia de comportamento diferente.

Essa variável permite ao modelo capturar padrões semanais específicos e melhorar a precisão das previsões.
