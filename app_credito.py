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
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
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
</style>
""", unsafe_allow_html=True)

# Título
st.title("🏦 Credit Intelligence System")
st.markdown("### AI-Powered Credit Risk Assessment Platform")
st.markdown("---")

# ============================================
# 1. GERAR DADOS SINTÉTICOS REALISTAS
# ============================================

@st.cache_data
def generate_realistic_data(n_samples=10000):
    """Gera dados sintéticos com distribuição realista de crédito"""
    np.random.seed(42)
    
    # Features com distribuições realistas
    income = np.random.normal(5000, 2500, n_samples)
    income = np.clip(income, 1200, 25000)
    
    # Dívida como porcentagem da renda (mais realista)
    debt_to_income_ratio = np.random.beta(2, 5, n_samples)  # Maioria com DTI baixo
    debt = income * debt_to_income_ratio
    
    employment = np.random.exponential(60, n_samples)
    employment = np.clip(employment, 0, 360)
    
    # Atrasos - maioria tem poucos ou nenhum
    late_payments = np.random.poisson(0.5, n_samples)
    late_payments = np.clip(late_payments, 0, 15)
    
    age = np.random.normal(42, 12, n_samples)
    age = np.clip(age, 22, 75)
    
    products = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
    
    # Variáveis categóricas com distribuição realista
    purpose_options = ['Personal', 'Property', 'Vehicle', 'Education']
    purpose_probs = [0.35, 0.25, 0.25, 0.15]
    purpose = np.random.choice(purpose_options, n_samples, p=purpose_probs)
    
    history_options = ['Good', 'Average', 'Poor']
    history_probs = [0.55, 0.30, 0.15]  # Maioria com histórico bom
    history = np.random.choice(history_options, n_samples, p=history_probs)
    
    own_property = np.random.choice(['Yes', 'No'], n_samples, p=[0.55, 0.45])
    savings = np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55])
    
    # Calcular SCORE DE CRÉDITO (0-1000) com pesos realistas
    # Começa com 700 (base média)
    credit_score = np.full(n_samples, 700, dtype=float)
    
    # Renda: + pontos para renda maior
    credit_score += (income / 5000) * 30
    credit_score = np.clip(credit_score, 0, 1000)
    
    # Dívida: - pontos para DTI alto
    dti = debt / income
    credit_score -= dti * 150
    
    # Emprego: + pontos para emprego longo
    credit_score += (employment / 120) * 40
    
    # Atrasos: penalização forte
    credit_score -= late_payments * 25
    
    # Idade: + estabilidade
    credit_score += ((age - 30) / 50) * 30 if age > 30 else 0
    
    # Produtos: diversificação é bom até certo ponto
    credit_score += min(products, 5) * 10
    
    # Histórico de crédito
    credit_score[history == 'Good'] += 80
    credit_score[history == 'Average'] += 30
    credit_score[history == 'Poor'] -= 50
    
    # Propriedade
    credit_score[own_property == 'Yes'] += 40
    
    # Poupança
    credit_score[savings == 'Yes'] += 35
    
    # Finalidade (algumas finalidades são mais seguras)
    credit_score[purpose == 'Property'] += 20
    credit_score[purpose == 'Education'] += 15
    credit_score[purpose == 'Vehicle'] += 5
    credit_score[purpose == 'Personal'] += 0
    
    # Adicionar ruído
    credit_score += np.random.normal(0, 20, n_samples)
    credit_score = np.clip(credit_score, 200, 850)
    
    # Calcular probabilidade de default (inversamente relacionada ao score)
    # Score 850 -> 1% de default, Score 300 -> 80% de default
    prob_default = 1 / (1 + np.exp((credit_score - 550) / 100))
    prob_default = np.clip(prob_default, 0.01, 0.85)
    
    # Default baseado na probabilidade
    default = (np.random.random(n_samples) < prob_default).astype(int)
    
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
        'credit_score': credit_score,
        'prob_default': prob_default,
        'default': default
    })
    
    return df

# ============================================
# 2. FUNÇÃO DE RATING BASEADA EM SCORE
# ============================================

def get_rating_from_score(credit_score):
    """Converte score de crédito em rating (baseado em agências reais)"""
    if credit_score >= 800:
        return "AAA", "Excelente", "#00ff00"
    elif credit_score >= 750:
        return "AA", "Muito Bom", "#88ff00"
    elif credit_score >= 700:
        return "A", "Bom", "#ccff00"
    elif credit_score >= 650:
        return "BBB", "Regular", "#ffcc00"
    elif credit_score >= 600:
        return "BB", "Atenção", "#ff8800"
    elif credit_score >= 550:
        return "B", "Risco Moderado", "#ff4400"
    elif credit_score >= 450:
        return "CCC", "Alto Risco", "#cc0000"
    else:
        return "D", "Crítico", "#8b0000"

def calculate_limit_from_income_and_rating(income, rating):
    """Calcula limite baseado em renda e rating"""
    base_limit = income * 3
    
    multipliers = {
        'AAA': 5.0,
        'AA': 3.5,
        'A': 2.5,
        'BBB': 1.8,
        'BB': 1.2,
        'B': 0.8,
        'CCC': 0.4,
        'D': 0.1
    }
    
    limit = base_limit * multipliers.get(rating, 0.5)
    return limit

def calculate_rate_from_rating(rating):
    """Calcula taxa baseada no rating"""
    rates = {
        'AAA': 0.99,
        'AA': 1.49,
        'A': 1.99,
        'BBB': 2.49,
        'BB': 3.49,
        'B': 4.99,
        'CCC': 7.99,
        'D': 15.99
    }
    return rates.get(rating, 5.99)

# ============================================
# 3. TREINAR MODELOS
# ============================================

@st.cache_resource
def train_models():
    """Treina modelos de machine learning"""
    
    df = generate_realistic_data()
    
    # Preparar features
    df_encoded = pd.get_dummies(df, columns=['purpose', 'credit_history', 'own_property', 'savings'])
    
    feature_cols = ['income', 'debt', 'employment_months', 'late_payments', 
                    'age', 'financial_products', 'dti'] + \
                   [col for col in df_encoded.columns if col.startswith(('purpose_', 'credit_history_', 'own_property_', 'savings_'))]
    
    X = df_encoded[feature_cols]
    y = df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    trained_models = {}
    metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        gini = 2 * auc - 1
        
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
    
    return trained_models, metrics, feature_cols, df

# ============================================
# 4. CALCULAR SCORE DO CLIENTE (SEM IA)
# ============================================

def calculate_client_score(client_data):
    """Calcula o score de crédito do cliente baseado nas informações"""
    
    score = 700  # Base
    
    # Renda
    income = client_data['income']
    score += (income / 5000) * 30
    
    # DTI (Debt-to-Income)
    dti = client_data['dti']
    score -= dti * 150
    
    # Tempo de emprego
    employment = client_data['employment_months']
    score += (employment / 120) * 40
    
    # Atrasos
    late = client_data['late_payments']
    score -= late * 25
    
    # Idade
    age = client_data['age']
    if age > 30:
        score += ((age - 30) / 50) * 30
    
    # Produtos
    products = client_data['financial_products']
    score += min(products, 5) * 10
    
    # Histórico
    history = client_data['credit_history']
    if history == 'Good':
        score += 80
    elif history == 'Average':
        score += 30
    else:
        score -= 50
    
    # Propriedade
    if client_data['own_property'] == 'Yes':
        score += 40
    
    # Poupança
    if client_data['savings'] == 'Yes':
        score += 35
    
    # Finalidade
    purpose = client_data['purpose']
    if purpose == 'Property':
        score += 20
    elif purpose == 'Education':
        score += 15
    elif purpose == 'Vehicle':
        score += 5
    
    # Garantir limites
    score = max(200, min(850, score))
    
    return score

# ============================================
# 5. INTERFACE PRINCIPAL
# ============================================

try:
    with st.spinner("🔄 Carregando sistema de IA..."):
        models, metrics, feature_cols, data = train_models()
    sistema_pronto = True
except Exception as e:
    st.error(f"Erro: {e}")
    sistema_pronto = False

if sistema_pronto:
    
    with st.sidebar:
        st.header("📋 Dados do Cliente")
        st.markdown("---")
        
        # Dados pessoais
        st.subheader("👤 Informações Pessoais")
        age = st.slider("Idade", 18, 80, 35)
        income = st.number_input("Renda Mensal (R$)", min_value=1000, max_value=50000, value=5000, step=500)
        
        st.subheader("💰 Situação Financeira")
        debt = st.number_input("Dívida Total (R$)", min_value=0, max_value=200000, value=5000, step=1000)
        
        if income > 0:
            dti = debt / income
            dti_percent = dti * 100
            if dti_percent < 20:
                st.success(f"✅ DTI: {dti_percent:.0f}% - Muito bom!")
            elif dti_percent < 35:
                st.info(f"📊 DTI: {dti_percent:.0f}% - Adequado")
            elif dti_percent < 50:
                st.warning(f"⚠️ DTI: {dti_percent:.0f}% - Atenção")
            else:
                st.error(f"🔴 DTI: {dti_percent:.0f}% - Alto risco")
        
        employment = st.slider("Tempo de Emprego (meses)", 0, 360, 36, 6)
        late_payments = st.slider("Pagamentos Atrasados (último ano)", 0, 20, 1)
        products = st.slider("Produtos Financeiros", 1, 10, 3)
        
        st.subheader("📋 Outras Informações")
        purpose = st.selectbox("Finalidade", ["Personal", "Property", "Vehicle", "Education"])
        credit_history = st.selectbox("Histórico de Crédito", ["Good", "Average", "Poor"], 
                                      help="Good = Sem restrições, Average = Alguns atrasos, Poor = Problemas sérios")
        own_property = st.selectbox("Possui Imóvel?", ["Yes", "No"])
        savings = st.selectbox("Possui Poupança/Investimentos?", ["Yes", "No"])
        
        st.markdown("---")
        
        use_ai = st.checkbox("🤖 Usar IA para análise avançada", value=True)
        
        if use_ai:
            model_choice = st.selectbox("Modelo de IA", list(models.keys()))
        
        analyze = st.button("🔍 ANALISAR CRÉDITO", use_container_width=True)
    
    # Área principal
    if analyze:
        
        # Calcular dados do cliente
        if income > 0:
            dti_calculado = debt / income
        else:
            dti_calculado = 0
        
        client_info = {
            'income': income,
            'debt': debt,
            'employment_months': employment,
            'late_payments': late_payments,
            'age': age,
            'financial_products': products,
            'purpose': purpose,
            'credit_history': credit_history,
            'own_property': own_property,
            'savings': savings,
            'dti': dti_calculado
        }
        
        # Calcular score do cliente (lógica explicável)
        credit_score = calculate_client_score(client_info)
        
        # Obter rating baseado no score
        rating, rating_desc, rating_color = get_rating_from_score(credit_score)
        
        # Calcular probabilidade de default baseada no score
        prob_default = 1 / (1 + np.exp((credit_score - 550) / 100))
        prob_default = min(0.85, max(0.01, prob_default))
        
        # Previsão da IA (se solicitado)
        if use_ai:
            try:
                # Preparar dados para IA
                df_pred = pd.DataFrame([client_info])
                df_pred_encoded = pd.get_dummies(df_pred)
                
                for col in feature_cols:
                    if col not in df_pred_encoded.columns:
                        df_pred_encoded[col] = 0
                
                df_pred_encoded = df_pred_encoded[feature_cols]
                
                ai_prob = models[model_choice].predict_proba(df_pred_encoded)[0, 1]
                
                # Média ponderada (60% score tradicional, 40% IA)
                final_prob = prob_default * 0.6 + ai_prob * 0.4
                
                st.info(f"🤖 O modelo {model_choice} estima {ai_prob*100:.1f}% de risco")
            except:
                final_prob = prob_default
                st.warning("IA temporariamente indisponível, usando análise tradicional")
        else:
            final_prob = prob_default
        
        # Calcular rating final
        if final_prob < 0.05:
            final_rating = "AAA"
            final_desc = "Excelente"
        elif final_prob < 0.10:
            final_rating = "AA"
            final_desc = "Muito Bom"
        elif final_prob < 0.18:
            final_rating = "A"
            final_desc = "Bom"
        elif final_prob < 0.28:
            final_rating = "BBB"
            final_desc = "Regular"
        elif final_prob < 0.40:
            final_rating = "BB"
            final_desc = "Atenção"
        elif final_prob < 0.55:
            final_rating = "B"
            final_desc = "Risco Moderado"
        elif final_prob < 0.75:
            final_rating = "CCC"
            final_desc = "Alto Risco"
        else:
            final_rating = "D"
            final_desc = "Crítico"
        
        # Calcular limite e taxa
        credit_limit = calculate_limit_from_income_and_rating(income, final_rating)
        interest_rate = calculate_rate_from_rating(final_rating)
        approval_prob = (1 - final_prob) * 100
        
        # EXIBIR RESULTADOS
        st.subheader("📊 RESULTADO DA ANÁLISE")
        st.markdown("---")
        
        # Cards principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Credit Score", f"{int(credit_score)}", 
                     delta=f"{credit_score - 550:+d}")
        
        with col2:
            st.metric("⭐ Rating", f"{final_rating} - {final_desc}")
        
        with col3:
            st.metric("⚠️ Risco de Default", f"{final_prob*100:.1f}%")
        
        with col4:
            st.metric("✅ Aprovação", f"{approval_prob:.0f}%")
        
        st.markdown("---")
        
        # Detalhes financeiros
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"💳 **Limite de Crédito Sugerido:** R$ {credit_limit:,.2f}")
            st.info(f"📉 **Taxa de Juros Estimada:** {interest_rate:.2f}% ao mês")
            
            if approval_prob >= 70:
                st.success(f"✅ **RECOMENDAÇÃO:** APROVAR - Cliente de baixo risco")
                st.balloons()
            elif approval_prob >= 50:
                st.warning(f"⚠️ **RECOMENDAÇÃO:** APROVAR COM GARANTIAS - Risco moderado")
            else:
                st.error(f"❌ **RECOMENDAÇÃO:** NEGAR - Alto risco de default")
        
        with col2:
            # Gráfico de score
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Barra de score
            score_normalizado = credit_score / 850
            cor = '#00ff00' if credit_score >= 600 else '#ff8800' if credit_score >= 450 else '#ff0000'
            
            ax.barh(['Score'], [score_normalizado * 100], color=cor, height=0.3)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Score (%)')
            ax.set_title(f'Score de Crédito: {int(credit_score)} / 850')
            
            # Adicionar marcadores de rating
            ax.axvline(x=800/850*100, color='darkgreen', linestyle='--', alpha=0.5, label='AAA')
            ax.axvline(x=700/850*100, color='green', linestyle='--', alpha=0.5, label='A')
            ax.axvline(x=600/850*100, color='orange', linestyle='--', alpha=0.5, label='BBB')
            ax.axvline(x=500/850*100, color='red', linestyle='--', alpha=0.5, label='CCC')
            
            ax.legend(loc='lower right')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # ============================================
        # MÉTRICAS DOS MODELOS
        # ============================================
        
        with st.expander("📊 Ver Performance dos Modelos de IA"):
            
            metrics_df = pd.DataFrame(metrics).T.round(4)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                metrics_df[['auc', 'gini', 'ks']].plot(kind='bar', ax=ax)
                ax.set_title('Métricas dos Modelos')
                ax.set_ylabel('Score')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.dataframe(metrics_df.style.background_gradient(cmap='RdYlGn', subset=['auc']))
        
        # ============================================
        # EXPLICAÇÃO DO SCORE
        # ============================================
        
        with st.expander("📖 Como o score é calculado"):
            st.markdown(f"""
            ### Seu Score: {int(credit_score)} / 850
            
            **Componentes do cálculo:**
            
            | Fator | Impacto no seu caso |
            |-------|---------------------|
            | Renda mensal | + {min(50, max(0, (income/5000)*30)):.0f} pontos |
            | Dívida/DTI | {min(100, max(-150, -dti_calculado*150)):.0f} pontos |
            | Tempo de emprego | + {min(40, (employment/120)*40):.0f} pontos |
            | Pagamentos atrasados | - {min(125, late_payments*25)} pontos |
            | Histórico de crédito | {80 if credit_history == 'Good' else 30 if credit_history == 'Average' else -50} pontos |
            | Possui imóvel | + {40 if own_property == 'Yes' else 0} pontos |
            | Possui poupança | + {35 if savings == 'Yes' else 0} pontos |
            | Finalidade do empréstimo | + {20 if purpose == 'Property' else 15 if purpose == 'Education' else 5 if purpose == 'Vehicle' else 0} pontos |
            
            **Classificação de Risco:**
            - **AAA (800-850):** Excelente - Risco mínimo
            - **AA (750-799):** Muito Bom - Risco muito baixo  
            - **A (700-749):** Bom - Risco baixo
            - **BBB (650-699):** Regular - Risco moderado
            - **BB (600-649):** Atenção - Risco elevado
            - **B (550-599):** Risco Moderado - Risco significativo
            - **CCC (450-549):** Alto Risco - Risco alto
            - **D (<450):** Crítico - Risco muito alto
            """)

else:
    st.error("Sistema não pôde ser carregado. Tente novamente.")

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <small>🏦 Credit Intelligence System | IA para Análise de Risco de Crédito</small>
</div>
""", unsafe_allow_html=True)
