# Previsão Interpretável de Vida Útil Remanescente (RUL) em Motores Turbofan

> **Projeto de Conclusão de Curso — MBA em Ciência de Dados e Inteligência Artificial (PUC Goiás, 2026)**  
> **Autor:** Danrley Nunes Montalvão  
> 📄 **Documentos:** [Artigo Completo em PDF](docs/artigo_rul_motores_turbofan.pdf) | [Slides da Apresentação](docs/apresentacao_slides_artigo_rul.pptx)

---

## 📌 Contexto e Problema de Negócio

Na manutenção preditiva de ativos críticos, a estimativa da **Vida Útil Remanescente (*Remaining Useful Life* - RUL)** indica quantos ciclos operacionais um equipamento ainda pode operar com segurança antes de falhar. 

Em operações industriais e aeronáuticas, o custo dos erros preditivos é altamente **assimétrico**:
* **Prever vida a menos (erro conservador):** Antecipa inspeções e descarta componentes ainda viáveis (impacto financeiro).
* **Prever vida a mais (erro tardio):** Mantém um equipamento degradado em operação, acarretando paradas não planejadas e riscos severos à segurança operacional.

Este projeto desenvolveu um pipeline ponta a ponta de *Machine Learning* sobre o benchmark **NASA C-MAPSS (subset FD001)** para comparar modelos preditivos clássicos sob três critérios: **precisão métrica, penalização de risco operacional e interpretabilidade física**.

---

## 🛠️ Pipeline Metodológico

NASA C-MAPSS (FD001)
│
├──> 1. Pré-processamento e Limpeza (Remoção de 7 sensores constantes)
├──> 2. Preparação do Alvo (RUL linear com teto de 125 ciclos no treino)
├──> 3. Engenharia de Atributos (14 sensores × 6 estatísticas rolantes = 84 atributos)
├──> 4. Validação Cruzada sem Vazamento (GroupKFold por motor)
├──> 5. Otimização Bayesiana de Hiperparâmetros (Optuna - 100 trials por modelo)
├──> 6. Avaliação Assimétrica (RMSE, MAE e PHM Score)
└──> 7. Interpretabilidade Global e Local (SHAP TreeExplainer)

### 1. Preparação dos Dados e Prevenção de Vazamento (*Data Leakage*)
* **Agrupamento estrito por motor (`unit_id`):** Ciclos de uma mesma unidade são dependentes temporalmente. Para impedir que ciclos do mesmo motor estivessem simultaneamente em treino e validação, a validação cruzada foi implementada com `GroupKFold` (5 folds).
* **Target Capping:** Aplicação de um teto de 125 ciclos apenas nos rótulos de treino, representando a fase inicial saudável (onde a degradação não é observável). O conjunto de teste manteve a distribuição real (sem teto).
* **Filtro de Sensores:** 7 sensores com variância nula ou quase nula foram descartados (sensores 1, 5, 6, 10, 16, 18 e 19), mantendo 14 sensores informativos.

### 2. Engenharia de Atributos (84 Features)
Para cada um dos 14 sensores informativos, foram geradas 6 estatísticas em janelas móveis de 30 ciclos (tamanho definido por análise de correlação de Spearman):
* Média móvel (`rolling_mean_30`)
* Desvio padrão móvel (`rolling_std_30`)
* Mínimo móvel (`rolling_min_30`)
* Máximo móvel (`rolling_max_30`)
* Inclinação linear móvel (`rolling_slope_30`)
* Diferença em relação ao nível basal inicial (`delta_baseline`)

### 3. Modelos Comparados
* **Baseline Linear:** Ridge Regression (regularização L2)
* **Bagging:** Random Forest
* **Boosting:** XGBoost e LightGBM
* **Meta-Ensemble:** Stacking (combinando RF + XGB + LGB com meta-modelo Ridge via previsões *out-of-fold*)

---

## 📊 Resultados e Comparação de Modelos

A avaliação no conjunto oficial de teste (100 motores isolados) avaliou métricas simétricas convencionais e a função de custo assimétrica **PHM Score**, que penaliza exponencialmente predições tardias (superestimação de vida útil).

| Modelo | CV RMSE | CV MAE | Teste RMSE | Teste MAE | PHM Score (Menor = Melhor) | Tamanho Serializado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge** | 16,81 ± 0,51 | 13,73 ± 0,69 | 18,59 | 14,85 | 501,2 | ~15 KB |
| **Random Forest** | 12,67 ± 1,02 | 9,11 ± 0,81 | 13,42 | 10,02 | 252,6 | ~45 MB |
| **XGBoost (Vencedor)** | **12,27 ± 0,67** | **8,70 ± 0,58** | **12,50** | **9,04** | **206,3** | **~840 KB** |
| **LightGBM** | 12,40 ± 0,78 | 8,73 ± 0,65 | 12,79 | 9,27 | 222,4 | ~1,2 MB |
| **Stacking** | 12,23 ± 0,70 | 8,55 ± 0,62 | 12,74 | 9,18 | 220,5 | ~81 MB |

### Principais Conclusões Técnicas:
1. **Desempenho do XGBoost:** Superou o baseline linear com uma redução de **32,8% no RMSE** e **58,8% no PHM Score**.
2. **Trade-off de Complexidade (Stacking vs. XGBoost):** Embora o Stacking tenha liderado marginalmente a validação cruzada, o XGBoost foi superior no teste com um artefato quase **100 vezes menor** (840 KB vs. 81 MB), tornando-o ideal para implantação em serviços de inferência e sistemas com restrição de memória.
3. **Distribuição do Erro:** O XGBoost apresentou viés médio de apenas -0,3 ciclo e nenhum erro tardio superior a 30 ciclos no conjunto de teste.

---

## 🔍 Interpretabilidade e Explicabilidade com SHAP

Para validar se o modelo estava aprendendo fenômenos físicos reais e evitar o comportamento de "caixa-preta", foi aplicado o **SHAP TreeExplainer**.

### Relevância Global dos Sensores
Os três sinais com maior impacto na previsão coincidem com subsistemas críticos de termodinâmica da turbina:
* **Sensor 11 (Ps30):** Pressão estática na saída do compressor de alta pressão (maior magnitude SHAP agregada).
* **Sensor 4 (T50):** Temperatura total na saída da turbina de baixa pressão.
* **Sensor 12 (Phi):** Razão entre fluxo de combustível e Ps30.

## 🚀 Como Executar o Projeto
Pré-requisitos
Python 3.10 ou superior
Gerenciador de ambiente virtual (venv ou conda)

### Instalação
1. Clone o repositório:

git clone [https://github.com/seu-usuario/turbofan-rul-prediction.git](https://github.com/seu-usuario/turbofan-rul-prediction.git)

cd turbofan-rul-prediction

2. Crie e ative um ambiente virtual:
python -m venv venv

venv\Scripts\activate

3. Instale as dependências:
pip install -r requirements.txt