# DataKLab - Analytics Platform

![](op.png)

A DataKLAB Analytics Platform é uma aplicação desktop desenvolvida em Python para análise de dados e machine learning. Ela oferece uma interface gráfica intuitiva para carregar, visualizar, analisar e modelar dados de diversos formatos.

## Recursos Principais

- **Carregamento de dados**: Suporta múltiplos formatos (CSV, Parquet, Excel, SAS)
- **Visualização de dados**: Exibição de dados em formato de tabela
- **Análise descritiva**: Geração de estatísticas descritivas e informações do dataset
- **Visualizações gráficas**: Criação de histogramas, boxplots, scatter plots, matriz de correlação e pairplots
- **Machine Learning**: Treinamento de modelo Perceptron para classificação
- **Salvamento de dados e modelos**: Exportação de datasets e modelos treinados

## Tecnologias Utilizadas

- Python 3.7+
- Bibliotecas:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - tkinter
  - pyreadstat (para arquivos SAS)
  - joblib

## Instalação

### Pré-requisitos

Certifique-se de ter o Python 3.7 ou superior instalado.

### Passos para instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/kauecodify/dataklab-analytics.git
   cd dataklab-analytics
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Como Executar

Após instalar as dependências, execute o aplicativo com:

```bash
python dataklab_app.py
```

## Guia de Uso

### Carregando Dados

1. Clique no botão "Carregar Dataset"
2. Selecione um arquivo de dados (CSV, Parquet, Excel, SAS)
3. O dataset será carregado e aparecerá no seletor "Dataset Ativo"

### Visualizando Dados

- Selecione um dataset no combobox
- Clique em "Visualizar Dados" para ver as primeiras 100 linhas

### Análise Descritiva

- Com um dataset carregado, clique em:
  - "Estatísticas Descritivas" para ver um resumo estatístico
  - "Informações do Dataset" para ver metadados

### Visualizações Gráficas

1. Selecione o tipo de gráfico desejado
2. Escolha as colunas para os eixos X e Y (quando aplicável)
3. Clique em "Gerar Gráfico"

### Machine Learning

1. Selecione a variável alvo
2. Clique em "Treinar Perceptron" para treinar um modelo
3. Use "Avaliar Modelo" para testar o desempenho
4. Salve o modelo treinado com "Salvar Modelo"

## Salvando Dados

- Use o botão "Salvar Dataset" para exportar o dataset ativo em:
  - Parquet
  - CSV
  - Excel
  - SAS
  - (Para Power BI, exporta como Parquet)

## Estrutura de Arquivos

- `dataklab_app.py` - Código principal da aplicação
- `requirements.txt` - Lista de dependências
- `dataklab_config.json` - Configuração persistente dos datasets carregados

## Contribuição

Contribuições são bem-vindas! Siga os passos:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Faça commit das suas alterações (`git commit -m 'Add some AmazingFeature'`)
4. Faça push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## Contato

Link do Projeto: [https://github.com/kauecodify/dataklab-analytics](https://github.com/kauecodify/dataklab_analytics)

---
