# Projeto Iris - Análise e Classificação
!pip install fpdf
# Este script apresenta um estudo sobre o famoso conjunto de dados Iris, incluindo uma análise exploratória, processamento e modelagem para classificação das espécies de flores.

# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from fpdf import FPDF
import os

# Definição da problemática
# O objetivo deste projeto é criar um modelo de machine learning capaz de classificar corretamente as espécies de flores Iris (Setosa, Versicolor, Virginica) com base em características morfológicas (comprimento e largura de pétalas e sépalas).
# Por que este problema é relevante?
# 1. Ele serve como benchmark para avaliar algoritmos de classificação.
# 2. Permite a aplicação prática de conceitos de machine learning em um conjunto de dados interpretável.

# Configurações iniciais e carregamento dos dados
def carregar_dados():
    # Carrega o conjunto de dados Iris
    return sns.load_dataset('iris')

data = carregar_dados()

# Análise Exploratória
def analise_exploratoria(dados):
    # Realiza estatísticas descritivas e visualizações
    print("\nEstatísticas descritivas:\n", dados.describe())

    # Pairplot
    sns.pairplot(dados, hue='species', diag_kind='kde')
    plt.show()

analise_exploratoria(data)

# Decisões de modelagem
# Baseado na análise exploratória, todas as características morfológicas serão utilizadas como variáveis preditoras. 
# Optamos por utilizar os seguintes algoritmos:
# 1. Logistic Regression - Simplicidade e interpretabilidade.
# 2. Random Forest - Robustez e habilidade de lidar com não linearidades.

# Implementação do modelo
def treinar_e_avaliar_modelos(dados):
    # Treina e avalia os modelos selecionados 
    X = dados.drop(columns='species')
    y = dados['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = classification_report(y_test, predictions, output_dict=True)
        print(f"\nModelo: {name}\n", classification_report(y_test, predictions))

    return results

resultados = treinar_e_avaliar_modelos(data)

# Salvar resultados como PDF
def salvar_como_pdf(data, resultados):
    # Criação do objeto PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Adicionando título e introdução
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Projeto Iris - Relatório", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Este relatório contém a análise exploratória, modelagem e avaliação do conjunto de dados Iris.")
    pdf.ln(10)

    # Estatísticas descritivas
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, txt="Estatísticas Descritivas", ln=True)
    pdf.set_font("Arial", size=12)
    stats = data.describe().to_string()
    pdf.multi_cell(0, 10, txt=stats)
    pdf.ln(10)

    # Resultados dos modelos
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, txt="Resultados dos Modelos", ln=True)
    pdf.set_font("Arial", size=12)

def salvar_como_pdf(data, resultados):

    for model_name, result in resultados.items():
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, txt=f"{model_name} - Relatório de Classificação", ln=True)
        pdf.set_font("Arial", size=10)
        
        # Corrected line: Accessing the classification report as a string directly
        classification_report_str = classification_report(y_test, model.predict(X_test))  
        
        pdf.multi_cell(0, 10, txt=classification_report_str)
        pdf.ln(10)

    # Salvar o PDF
    output_path = "relatorio_projeto_iris.pdf"
    pdf.output(output_path)

    return output_path

# Chamar a função para salvar o relatório como PDF
salvar_como_pdf(data, resultados)

def upload_para_github():
    # Faz upload do relatório para o GitHub 
    repo_url = "https://github.com/GuiMiranda123/Projeto-Iris-Melhorias.git"
    token = "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"  # Substitua pelo seu token de acesso pessoal

    # Configurações do Git
    !git init # Indentation corrected
    !git config user.name 'GuiMiranda123' # Indentation corrected
    !git config user.email 'miranda.guiferreira@' # Indentation corrected
    !git remote add origin https://{token}@{repo_url.split('https://')[1]} # Indentation corrected

    # Commit e push dos arquivos
    !git add . # Indentation corrected
    !git commit -m 'Adicionando relatório do projeto Iris' # Indentation corrected
    !git branch -M main # Indentation corrected
    !git push -u origin main # Indentation corrected
    print("Upload concluído para o GitHub!")
