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

## Colunas Necessárias:

- distance:
    - Indica a distancia capturada pelo sensor (abaixo de 40 cm pode indicar que o carro esta presente na vaga)

- light:
    - Indica a luz capturada pelo sensor, nesse caso, valores mais elevados significam ambientes mais escuros e valores mais baixos (+ próximos de 0) indicam ambientes iluminados.

- is_available:
    - Se o valor de light estiver acima de 600 e da distance estiver abaixo de 40, is_available será 0 (indicando que possui carro na vaga) , caso o valor de is_available seja 1, indica que a vaga esta disponível (ou seja, luminosidade boa,  e distancia maior que 40) **OSB: Para uma vaga ser considerada ocupada ambos os parametros (light e distance) devem atender os requisitos fornecidos**. 

- Dia:
    - Indica qual dia foi ocupada a vaga no formato DD/MM/AAAA 

- Horário Entrada e Saída:
    - Indica o horáro de entrada e saída de veiculos de uma vaga. Vagas que não tiverem os horários cadastrados indica que estavam desocupadas. Por exemplo: Vaga A1: d=30, l=800, isava=0 -> nesse caso, vai possuir um horário de entrada (ocupação da vaga por um veículo) e saída. Formato vai ser: HH:MM:SS (Exemplos gerados o SS vai ser sempre 0)

- Vaga:
    - Indica a vaga com seus demais dados. Ex: Vaga A1:


Sabendo que o estacionamento ficará aberto de segunda a segunda das 6hrs da manhã até as 22hrs da noite, para randomizarmos os horários geraremos valores floats com 2 digitos entre 6 e 22. Esses valores representam os nossos horários e a % da dos minutos de uma hora, por exemplo:

| Valor (float)     | Calc min         | Resultado em horas         |
|-------------------|------------------|----------------------------|
| 8.20              | 60(min) * 0,2   | 8hrs e 12min                |
| 20.50             | 60 * 0,5        | 20 horas 30min              |
| 5400              | 5400 / 3600     | 1,5 horas (1 hora e 30 min) |


48 vagas -> A1 -> 21600 -> 22400 -> ficou 12min
6-22 ->

A1 -> is_available = 0 (vaga ocupada)
    -> distance <= 40 (random int(entre 0-40))
    -> light >= 600 (random int(entre 600-900))
    -> Tem um hora_Ini -> hora_fim


| Vaga || is_available | distance | light | | DIA/MES/ANO | | Hora entrada | | Hora Saída | | Tempo Total Ocupado |

1 (A1) vaga -> pode ser ocupada 06:00 -> 06:30 -> 06:40-6:50 -> 7:00 -> 7:45...
A1 6 - 22

for vaga in range(48):
    random(float(6-22)) -> 6.50 -> 06:30 -> 60 * 0,5 

Precisa de um range para o numero de dias, exemplo:

dias = 100
data_ini = 21/10/2023 -> 100 

for dia in range(dias): # seriam gerados dados para 100 dias
    for vaga in range(48): # Para cada seria gerado os dados para as vagas
        array = [['hora_ini', 'hora_fim'], ['hora_ini', 'hora_fim'], ['hora_ini', 'hora_fim']]

        for batata in array:
            random(float(6-22)) randint(hora_Ini)
            hora_ini = random(batata[0], batata[1])
            hora_fim = random(hora_Ini, batata[1])
            adiciona no excel

ficou livre 06:30 -> 06:40