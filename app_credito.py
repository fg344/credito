import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, log_loss)
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Credit Intelligence System",
    page_icon="🏦",
    layout="wide"
)

# Estilo customizado
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #36ff9f);
        color: #08111f;
        font-weight: bold;
        border-radius: 20px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.title("🏦 Credit Intelligence System")
st.markdown("### AI-Powered Credit Risk Assessment Platform")
st.markdown("---")

# ============================================
# 1. GERAR DADOS SINTÉTICOS PARA TREINAMENTO
# ============================================

@st.cache_data
def generate_synthetic_data(n_samples=10000):
    """Gera dados sintéticos realistas para análise de crédito"""
    np.random.seed(42)
    
    # Features
    income = np.random.normal(5000, 2000, n_samples)
    income = np.clip(income, 1000, 15000)
    
    debt = income * np.random.uniform(0.1, 0.8, n_samples)
    employment = np.random.exponential(48, n_samples)
    employment = np.clip(employment, 0, 240)
    
    late_payments = np.random.poisson(1, n_samples)
    late_payments = np.clip(late_payments, 0, 20)
    
    age = np.random.normal(40, 12, n_samples)
    age = np.clip(age, 18, 80)
    
    products = np.random.randint(1, 8, n_samples)
    
    # Variáveis categóricas
    purpose_map = {'Personal': 0.3, 'Property': 0.25, 'Vehicle': 0.25, 'Education': 0.2}
    history_map = {'Good': 0.7, 'Average': 0.2, 'Poor': 0.1}
    property_map = {'Yes': 0.6, 'No': 0.4}
    savings_map = {'Yes': 0.55, 'No': 0.45}
    
    purpose = np.random.choice(list(purpose_map.keys()), n_samples, p=list(purpose_map.values()))
    history = np.random.choice(list(history_map.keys()), n_samples, p=list(history_map.values()))
    own_property = np.random.choice(list(property_map.keys()), n_samples, p=list(property_map.values()))
    savings = np.random.choice(list(savings_map.keys()), n_samples, p=list(savings_map.values()))
    
    # Calcular score de crédito (target)
    dti = debt / income  # Debt-to-income ratio
    
    score_raw = (
        0.3 * (income / 10000) +
        -0.5 * dti +
        0.2 * (employment / 120) +
        -0.4 * (late_payments / 10) +
        0.15 * (age / 80) +
        0.1 * (products / 7) +
        (1 if history == 'Good' else 0.5 if history == 'Average' else 0) +
        (0.2 if own_property == 'Yes' else 0) +
        (0.2 if savings == 'Yes' else 0)
    )
    
    # Adicionar ruído
    score_raw += np.random.normal(0, 0.1, n_samples)
    
    # Probabilidade de default
    prob_default = 1 / (1 + np.exp(-(score_raw * 2 - 1)))
    
    # Target (default = 1, good = 0)
    default = (prob_default > 0.5).astype(int)
    
    # Criar DataFrame
    df = pd.DataFrame({
        'income': income,
        'debt': debt,
        'employment_months': employment,
        'late_payments': late_payments,
        'age': age,
        'financial_products': products,
        'purpose': purpose,
        'credit_history': history,
        'own_property': own_property,
        'savings': savings,
        'dti': dti,
        'prob_default': prob_default,
        'default': default
    })
    
    return df

# ============================================
# 2. TREINAR MODELOS
# ============================================

@st.cache_resource
def train_models():
    """Treina múltiplos modelos de machine learning"""
    
    # Carregar dados
    df = generate_synthetic_data()
    
    # Preparar features
    feature_cols = ['income', 'debt', 'employment_months', 'late_payments', 
                    'age', 'financial_products', 'dti']
    
    # Features categóricas codificadas
    df_encoded = pd.get_dummies(df, columns=['purpose', 'credit_history', 
                                               'own_property', 'savings'])
    
    X = df_encoded[['income', 'debt', 'employment_months', 'late_payments', 
                    'age', 'financial_products', 'dti',
                    'purpose_Education', 'purpose_Personal', 'purpose_Property', 'purpose_Vehicle',
                    'credit_history_Average', 'credit_history_Good', 'credit_history_Poor',
                    'own_property_No', 'own_property_Yes',
                    'savings_No', 'savings_Yes']]
    
    y = df['default']
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Modelos
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    trained_models = {}
    metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
        auc = roc_auc_score(y_test, y_pred_proba)
        gini = 2 * auc - 1
        
        # KS Statistic
        ks_stat, _ = ks_2samp(y_pred_proba[y_test == 0], y_pred_proba[y_test == 1])
        
        metrics[name] = {
            'auc': auc,
            'gini': gini,
            'ks': ks_stat,
            'accuracy': accuracy_score(y_test, model.predict(X_test)),
            'precision': precision_score(y_test, model.predict(X_test)),
            'recall': recall_score(y_test, model.predict(X_test)),
            'f1': f1_score(y_test, model.predict(X_test))
        }
        
        trained_models[name] = model
    
    return trained_models, metrics, X_train.columns, df

# ============================================
# 3. FUNÇÃO DE PREDIÇÃO
# ============================================

def predict_credit(client_data, model, feature_columns):
    """Faz predição para um novo cliente"""
    
    # Criar DataFrame com todas as features
    df_pred = pd.DataFrame([client_data])
    
    # One-hot encoding
    df_pred_encoded = pd.get_dummies(df_pred)
    
    # Garantir que tem todas as colunas
    for col in feature_columns:
        if col not in df_pred_encoded.columns:
            df_pred_encoded[col] = 0
    
    df_pred_encoded = df_pred_encoded[feature_columns]
    
    # Predição
    prob_default = model.predict_proba(df_pred_encoded)[0, 1]
    rating = get_rating(prob_default)
    
    return prob_default, rating

def get_rating(prob_default):
    """Converte probabilidade em rating"""
    if prob_default < 0.02:
        return "AAA"
    elif prob_default < 0.05:
        return "AA"
    elif prob_default < 0.10:
        return "A"
    elif prob_default < 0.20:
        return "BBB"
    elif prob_default < 0.35:
        return "BB"
    elif prob_default < 0.50:
        return "B"
    elif prob_default < 0.70:
        return "CCC"
    else:
        return "D"

def calculate_limit(income, rating):
    """Calcula limite de crédito baseado em renda e rating"""
    base_limit = income * 5
    
    rating_multipliers = {
        'AAA': 2.0,
        'AA': 1.5,
        'A': 1.2,
        'BBB': 1.0,
        'BB': 0.7,
        'B': 0.5,
        'CCC': 0.3,
        'D': 0.1
    }
    
    limit = base_limit * rating_multipliers.get(rating, 0.5)
    return limit

def calculate_interest_rate(rating, prob_default):
    """Calcula taxa de juros baseada no rating"""
    base_rates = {
        'AAA': 0.8,
        'AA': 1.2,
        'A': 1.8,
        'BBB': 2.5,
        'BB': 3.5,
        'B': 5.0,
        'CCC': 8.0,
        'D': 15.0
    }
    
    rate = base_rates.get(rating, 5.0) + (prob_default * 10)
    return min(rate, 25.0)

# ============================================
# 4. INTERFACE DO USUÁRIO
# ============================================

# Treinar modelos
with st.spinner("🔄 Treinando modelos de IA..."):
    models, metrics, feature_cols, data = train_models()

# Sidebar - Input do cliente
with st.sidebar:
    st.header("📋 Dados do Cliente")
    st.markdown("---")
    
    # Dados pessoais
    st.subheader("Informações Pessoais")
    age = st.number_input("Idade", min_value=18, max_value=100, value=35)
    income = st.number_input("Renda Mensal (R$)", min_value=1000, max_value=50000, value=5000)
    
    # Dados financeiros
    st.subheader("Situação Financeira")
    debt = st.number_input("Dívida Total (R$)", min_value=0, max_value=500000, value=8000)
    dti = debt / income if income > 0 else 0
    st.caption(f"💰 Debt-to-Income Ratio: {dti:.2%}")
    
    employment = st.number_input("Tempo de Emprego (meses)", min_value=0, max_value=480, value=24)
    late_payments = st.number_input("Pagamentos Atrasados (últimos 12 meses)", min_value=0, max_value=50, value=2)
    products = st.number_input("Produtos Financeiros", min_value=1, max_value=15, value=3)
    
    # Variáveis categóricas
    st.subheader("Outras Informações")
    purpose = st.selectbox("Finalidade do Empréstimo", 
                          ["Personal", "Property", "Vehicle", "Education"])
    credit_history = st.selectbox("Histórico de Crédito", 
                                  ["Good", "Average", "Poor"])
    own_property = st.selectbox("Possui Imóvel?", ["Yes", "No"])
    savings = st.selectbox("Possui Poupança?", ["Yes", "No"])
    
    st.markdown("---")
    
    # Seleção do modelo
    model_choice = st.selectbox("Modelo de IA para Análise", 
                                list(models.keys()))
    
    analyze_button = st.button("🔍 ANALISAR CRÉDITO", use_container_width=True)

# Área principal
if analyze_button:
    
    # Preparar dados do cliente
    client_info = {
        'income': income,
        'debt': debt,
        'employment_months': employment,
        'late_payments': late_payments,
        'age': age,
        'financial_products': products,
        'dti': dti,
        'purpose': purpose,
        'credit_history': credit_history,
        'own_property': own_property,
        'savings': savings
    }
    
    # Fazer predição
    with st.spinner("🧠 Processando análise..."):
        prob_default, rating = predict_credit(client_info, models[model_choice], feature_cols)
    
    # Calcular limites e taxas
    credit_limit = calculate_limit(income, rating)
    interest_rate = calculate_interest_rate(rating, prob_default)
    approval_prob = (1 - prob_default) * 100
    
    # Layout de resultados
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Credit Score", f"{int((1-prob_default)*1000)}", 
                 delta=f"Rating {rating}")
    
    with col2:
        st.metric("⚠️ Probabilidade de Default", f"{prob_default*100:.1f}%",
                 delta="Risco" if prob_default > 0.2 else "Baixo Risco")
    
    with col3:
        st.metric("✅ Aprovação", f"{approval_prob:.1f}%",
                 delta="Aprovado" if approval_prob > 50 else "Negado")
    
    with col4:
        st.metric("💳 Limite Sugerido", f"R$ {credit_limit:,.2f}")
    
    st.markdown("---")
    
    # Taxa e detalhes
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"📊 **Taxa de Juros Estimada:** {interest_rate:.2f}% ao mês")
        st.info(f"🏷️ **Classificação de Risco:** {rating}")
        
        # Recomendação
        if prob_default < 0.15:
            st.success("✅ **Recomendação:** APROVAR - Cliente de baixo risco")
        elif prob_default < 0.35:
            st.warning("⚠️ **Recomendação:** APROVAR COM GARANTIAS - Risco moderado")
        else:
            st.error("❌ **Recomendação:** NEGAR - Alto risco de default")
    
    with col2:
        # Gráfico de risco
        fig, ax = plt.subplots(figsize=(8, 4))
        risk_levels = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
        risk_colors = ['#00ff88', '#88ff00', '#ffff00', '#ffcc00', '#ff8800', '#ff4400', '#cc0000', '#8b0000']
        
        current_risk_index = risk_levels.index(rating) if rating in risk_levels else 4
        
        bars = ax.barh(risk_levels, range(len(risk_levels), 0, -1), 
                      color=risk_colors, alpha=0.7)
        bars[current_risk_index].set_color('#ff4444')
        ax.axvline(x=len(risk_levels) - current_risk_index, color='blue', 
                  linestyle='--', linewidth=2, label='Cliente')
        ax.set_xlabel('Nível de Risco (maior = melhor)')
        ax.set_title('Posicionamento de Risco')
        ax.legend()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # ============================================
    # 5. MÉTRICAS DO MODELO E BACKTESTING
    # ============================================
    
    st.subheader("📊 Performance dos Modelos de IA")
    
    # Criar DataFrame com métricas
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.round(4)
    
    # Gráfico comparativo
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_to_plot = ['auc', 'gini', 'ks']
        metrics_df[metrics_to_plot].plot(kind='bar', ax=ax)
        ax.set_title('Métricas de Avaliação dos Modelos')
        ax.set_ylabel('Score')
        ax.set_xlabel('Modelo')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Tabela completa
        st.dataframe(metrics_df.style.background_gradient(cmap='RdYlGn', subset=['auc', 'gini', 'ks']))
        
        # Explicação das métricas
        with st.expander("📖 Entendendo as Métricas"):
            st.markdown("""
            - **KS (Kolmogorov-Smirnov):** Mede a capacidade de separação entre bons e maus pagadores. > 0.3 é bom.
            - **ROC/AUC:** Área sob a curva ROC. > 0.7 é aceitável, > 0.8 é excelente.
            - **Gini:** 2*AUC - 1. Mede a concentração de risco.
            - **Backtesting:** Validação cruzada com 5 folds para garantir robustez.
            """)
    
    st.markdown("---")
    
    # ============================================
    # 6. BACKTESTING E VALIDAÇÃO
    # ============================================
    
    st.subheader("🔄 Backtesting e Validação do Modelo")
    
    # Realizar backtesting (validação cruzada)
    model_obj = models[model_choice]
    
    # Preparar dados completos
    data_encoded = pd.get_dummies(data, columns=['purpose', 'credit_history', 'own_property', 'savings'])
    X_full = data_encoded[feature_cols]
    y_full = data['default']
    
    # Cross-validation
    cv_scores_auc = cross_val_score(model_obj, X_full, y_full, cv=5, scoring='roc_auc')
    cv_scores_accuracy = cross_val_score(model_obj, X_full, y_full, cv=5, scoring='accuracy')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📈 AUC Médio (5-fold)", f"{cv_scores_auc.mean():.3f}",
                 delta=f"±{cv_scores_auc.std():.3f}")
    
    with col2:
        st.metric("🎯 Acurácia Média", f"{cv_scores_accuracy.mean():.2%}",
                 delta=f"±{cv_scores_accuracy.std():.2%}")
    
    with col3:
        # Estabilidade do modelo
        stability = 1 - cv_scores_auc.std()
        st.metric("🔒 Estabilidade", f"{stability:.2%}",
                 delta="Robusto" if stability > 0.95 else "Variável")
    
    # Gráfico de validação cruzada
    fig, ax = plt.subplots(figsize=(10, 5))
    folds = range(1, 6)
    ax.plot(folds, cv_scores_auc, 'o-', linewidth=2, markersize=8, label='AUC')
    ax.axhline(y=cv_scores_auc.mean(), color='r', linestyle='--', label=f'Média = {cv_scores_auc.mean():.3f}')
    ax.fill_between(folds, cv_scores_auc.mean() - cv_scores_auc.std(), 
                    cv_scores_auc.mean() + cv_scores_auc.std(), alpha=0.2)
    ax.set_xlabel('Fold')
    ax.set_ylabel('AUC Score')
    ax.set_title('Validação Cruzada - Performance por Fold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("---")
    
    # ============================================
    # 7. CURVA ROC
    # ============================================
    
    st.subheader("📈 Curva ROC - Capacidade de Discriminação do Modelo")
    
    # Calcular ROC para o modelo selecionado
    data_encoded = pd.get_dummies(data, columns=['purpose', 'credit_history', 'own_property', 'savings'])
    X_full = data_encoded[feature_cols]
    y_full = data['default']
    
    y_pred_proba = model_obj.predict_proba(X_full)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_full, y_pred_proba)
    auc_score = roc_auc_score(y_full, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.2)
    ax.set_xlabel('False Positive Rate (1 - Specificidade)')
    ax.set_ylabel('True Positive Rate (Sensibilidade)')
    ax.set_title(f'Curva ROC - {model_choice}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Adicionar ponto ótimo
    optimal_idx = np.argmax(tpr - fpr)
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
           label=f'Ponto Ótimo (Threshold = {thresholds[optimal_idx]:.2f})')
    ax.legend()
    
    st.pyplot(fig)
    
    # Interpretação
    if auc_score > 0.8:
        st.success(f"✅ **Excelente!** O modelo {model_choice} tem excelente capacidade de discriminação (AUC = {auc_score:.3f})")
    elif auc_score > 0.7:
        st.info(f"📊 **Bom!** O modelo {model_choice} tem boa capacidade de discriminação (AUC = {auc_score:.3f})")
    else:
        st.warning(f"⚠️ **Atenção!** O modelo {model_choice} pode precisar de ajustes (AUC = {auc_score:.3f})")

else:
    # Tela inicial
    st.info("👈 **Preencha os dados do cliente na barra lateral e clique em ANALISAR CRÉDITO**")
    
    # Mostrar exemplo
    with st.expander("📖 Sobre o Sistema"):
        st.markdown("""
        ### Credit Intelligence System
        
        **Modelos implementados:**
        - ✅ Regressão Logística
        - ✅ Random Forest
        - ✅ Gradient Boosting
        - ✅ SVM
        - ✅ Redes Neurais
        
        **Métricas de avaliação:**
        - 📊 KS (Kolmogorov-Smirnov)
        - 📈 ROC / AUC
        - 🎯 Gini Coefficient
        - 🔄 Backtesting com 5-fold CV
        
        **O que o sistema faz:**
        1. Analisa o perfil financeiro do cliente
        2. Calcula probabilidade de default
        3. Atribui rating de crédito (AAA a D)
        4. Sugere limite e taxa de juros
        5. Valida modelo com backtesting
        """)
    
    # Dashboard de performance dos modelos
    st.subheader("📊 Performance dos Modelos (Benchmark)")
    
    # Criar gráfico comparativo de todos os modelos
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = list(metrics.keys())
    auc_scores = [metrics[m]['auc'] for m in model_names]
    gini_scores = [metrics[m]['gini'] for m in model_names]
    ks_scores = [metrics[m]['ks'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax.bar(x - width, auc_scores, width, label='AUC', color='#2ecc71')
    ax.bar(x, gini_scores, width, label='Gini', color='#3498db')
    ax.bar(x + width, ks_scores, width, label='KS', color='#e74c3c')
    
    ax.set_xlabel('Modelos')
    ax.set_ylabel('Score')
    ax.set_title('Comparação de Performance entre Modelos')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <small>🏦 Credit Intelligence System | Machine Learning Credit Risk Analysis | Powered by AI</small>
</div>
""", unsafe_allow_html=True)
