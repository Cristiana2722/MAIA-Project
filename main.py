import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# Configurare pagină
st.set_page_config(
    page_title="Tema MAIA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizat
st.markdown("""
<style>
    .stApp {
        background-color: #706e9e;
        color: #white;
    }

    [data-testid="stSidebar"] {
        background-color: #4d4a85;
    }
    
    header { 
        background-color: #29266d !important; 
    }

    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #b8b7ce;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #dbdbe7;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        color: black;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data_from_file(uploaded_file):
    """Încarcă datele dintr-un fișier CSV sau Excel."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Sunt acceptate doar fișiere de tip CSV sau Excel.")

    return df


# Sidebar navigation
def sidebar_navigation():
    st.sidebar.markdown("#  Temă Metode și tehnici avansate de inteligență artificială")

    sections = [
        " Încărcarea Datelor",
        " Filtrarea Datelor",
        " Explorarea Datelor",
        " Analiza Coloanelor Numerice",
        " Analiza Coloanelor Categorice",
        " Corelații și Outlieri"
    ]

    selected = st.sidebar.radio("Meniu:", sections)

    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Informații")
    st.sidebar.info(
        "Temă realizată de: COSTAN Cristiana\n\n"
        "Anul 2, Grupa 1126, BDSA\n\n"
    )

    return selected


# Cerința 1
def upload_files():
    st.markdown('<h1 class="main-header">Încărcarea Datelor</h1>', unsafe_allow_html=True)

    with st.expander("Cerințe abordate în această secțiune", expanded=False):
        st.markdown("""
        • Încărcare fișier CSV sau Excel\n
        • Validare că fișierul a fost citit corect\n
        • Afișare mesaj de confirmare\n
        • Afișare primele 10 rânduri din dataset\n
        """)

    with st.expander("Despre setul de date utilizat", expanded=False):
        st.markdown("""
        ### Sursă set de date:
        ***Titanic Dataset***
        ```
        https://www.kaggle.com/datasets/yasserh/titanic-dataset
        ```
        """)

    st.markdown('<div class="sub-header">Încarcă datele dintr-un fișier CSV sau Excel</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Alege fișierul:", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        try:
            df_file = load_data_from_file(uploaded_file)
            st.session_state['df'] = df_file
            file_name = uploaded_file.name
            st.session_state['dataset_name'] = file_name

            st.success(f"Date încărcate cu succes! ({len(df_file):,} rânduri, {len(df_file.columns)} coloane)")

            st.dataframe(df_file.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Eroare la citirea fișierului: {e}")
    else:
        st.info(" Încarcă un fișier CSV sau Excel pentru a începe!")


# Cerința 1 - filtrare
def filter_dataset():
    st.markdown('<h1 class="main-header">Filtrarea Datelor</h1>', unsafe_allow_html=True)

    with st.expander("Cerințe abordate în această secțiune", expanded=False):
        st.markdown("""
        • Creare slidere pentru filtrare coloane numerice\n
        • Creare multiselect pentru filtrare coloane categorice\n
        • Afișare număr rânduri înainte și după filtrare\n
        • Afișare dataframe filtrat\n
        """)

    if 'df' not in st.session_state:
        st.warning("Te rog să încarci mai întâi datele din secțiunea 'Încărcarea Datelor'!")
        return

    df = st.session_state['df'].copy()

    st.markdown('<div class="sub-header">Filtrarea datelor încărcate</div>', unsafe_allow_html=True)

    df_filtered = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        st.markdown("**Filtrare coloane numerice**")
        for col in numeric_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            selected_range = st.slider(
                f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            df_filtered = df_filtered[
                (df_filtered[col] >= selected_range[0]) &
                (df_filtered[col] <= selected_range[1])
                ]

        categorical_cols = df.select_dtypes(exclude="number").columns
        if len(categorical_cols) > 0:
            st.markdown("**Filtrare coloane categorice**")
            for col in categorical_cols:
                options = df[col].dropna().unique().tolist()
                selected_options = st.multiselect(
                    f"{col}",
                    options,
                    default=options
                )
                if selected_options:
                    df_filtered = df_filtered[df_filtered[col].isin(selected_options)]

        st.subheader(" Rezumat filtrare")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rânduri inițiale", df.shape[0])
        with col2:
            st.metric("Rânduri după filtrare", df_filtered.shape[0])

        st.subheader(" Setul de date filtrat")
        st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)

    else:
        st.info(" Încarcă un fișier CSV sau Excel pentru a începe!")


# Cerința 2
def dataset_info():
    st.markdown('<h1 class="main-header">Filtrarea Datelor</h1>', unsafe_allow_html=True)

    with st.expander("Cerințe abordate în această secțiune", expanded=False):
        st.markdown("""
        • Afișare număr rânduri și coloane\n
        • Afișare tipuri de date pentru fiecare coloană\n
        • Identificare coloane cu valori lipsă\n
        • Calcul procent valori lipsă per coloană\n
        • Creare grafic pentru vizualizarea valorilor lipsă\n
        • Afișare statistici descriptive pentru coloane numerice (mean, median, std, min, max, quartile)\n
        """)

    if 'df' not in st.session_state:
        st.warning("Te rog să încarci mai întâi datele din secțiunea 'Încărcarea Datelor'!")
        return

    df = st.session_state['df'].copy()

    st.markdown('<div class="sub-header">Explorarea datelor încărcate</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total rânduri", f"{len(df):,}")

    with col2:
        st.metric("Total coloane", len(df.columns))

    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric(" Memorie", f"{memory_mb:.2f} MB")

    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric(" Valori lipsă", f"{missing_pct:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs([" Preview", " Info", " Statistici", " Vizualizare"])

    with tab1:
        st.markdown("### Primele rânduri")
        n_rows = st.slider("Număr rânduri de afișat:", 5, 50, 10, key="preview_rows")
        st.dataframe(df.head(n_rows), use_container_width=True)

        with st.expander("Ultimele rânduri"):
            st.dataframe(df.tail(n_rows), use_container_width=True)

    with tab2:
        st.markdown("### Informații dataset")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Tipuri de date:**")
            dtype_df = pd.DataFrame({
                'Coloană': df.columns,
                'Tip': df.dtypes.astype(str),
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True)

        with col2:
            st.markdown("**Distribuția tipurilor de date:**")
            type_counts = df.dtypes.astype(str).value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Tipuri de date"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Statistici descriptive")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown("**Coloane numerice:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.markdown("**Coloane categorice:**")
            cat_summary = pd.DataFrame({
                col: [
                    df[col].nunique(),
                    df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                    df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
                    f"{(df[col].value_counts().iloc[0] / len(df) * 100):.1f}%" if len(df[col]) > 0 else "0%"
                ] for col in categorical_cols
            }, index=['Valori unice', 'Cel mai comun', 'Frecvență', 'Procent']).T
            st.dataframe(cat_summary, use_container_width=True)

    with tab4:
        st.markdown("### Vizualizare valori lipsă")

        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Coloană': missing.index,
            'Valori lipsă': missing.values,
            'Procent valori lipsă': missing_pct.values
        }).sort_values('Valori lipsă', ascending=False)

        cols_with_missing = missing_df[missing_df['Valori lipsă'] > 0]

        if len(cols_with_missing) > 0:
            fig = px.bar(
                cols_with_missing,
                x='Coloană',
                y='Procent valori lipsă',
                title='Procentul valorilor lipsă pe coloană',
                text='Valori lipsă'
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(cols_with_missing, use_container_width=True)
        else:
            st.success(" Nu există valori lipsă în dataset!")

        if len(cols_with_missing) > 0:
            st.markdown("### Heatmap valori lipsă (primele 50 rânduri)")
            colours = ['#ff9999', '#af69ee']  # pink = missing, purple = present
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df.head(50).isnull(), cmap=sns.color_palette(colours),
                        cbar=False, yticklabels=False, ax=ax)
            ax.set_title("Roz = Lipsă, Mov = Prezent")
            st.pyplot(fig)

        if len(cols_with_missing) > 0:
            st.markdown("### Heatmap valori lipsă (setul de date întreg)")
            colours = ['#ff9999', '#af69ee']  # pink = missing, purple = present
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df.isnull(), cmap=sns.color_palette(colours),
                        cbar=False, yticklabels=False, ax=ax)
            ax.set_title("Roz = Lipsă, Mov = Prezent")
            st.pyplot(fig)


# Cerința 3
def numeric_values():
    st.markdown('<h1 class="main-header">Analiza Coloanelor Numerice</h1>', unsafe_allow_html=True)

    with st.expander("Cerințe abordate în această secțiune", expanded=False):
        st.markdown("""
        • Selectare coloană numerică de către utilizator\n
        • Creare histogram interactiv\n
        • Adăugare slider pentru numărul de bins (10-100)\n
        • Creare box plot pentru aceeași coloană\n
        • Calcul și afișare: medie, mediană, deviație standard\n
        """)

    if 'df' not in st.session_state:
        st.warning("Te rog să încarci mai întâi datele din secțiunea 'Încărcarea Datelor'!")
        return

    df = st.session_state['df'].copy()

    st.markdown('<div class="sub-header">Analiza coloanelor numerice</div>', unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("Datasetul nu conține coloane numerice.")
    else:
        selected_col = st.selectbox(
            "Selectează o coloană numerică:",
            numeric_cols
        )

        col_data = df[selected_col].dropna()

        if col_data.empty:
            st.warning("Coloana selectată nu conține valori numerice valide.")
        else:
            bins = st.slider(
                "Număr de bins pentru histogramă:",
                min_value=10,
                max_value=100,
                value=30
            )

            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()

            st.markdown('<div class="sub-header">Statistici descriptive</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Medie", f"{mean_val:.2f}")
            col2.metric("Mediană", f"{median_val:.2f}")
            col3.metric("Deviație standard", f"{std_val:.2f}")

            st.markdown('<div class="sub-header">Histogramă</div>', unsafe_allow_html=True)
            fig_hist = px.histogram(
                col_data,
                nbins=bins,
                title=f"Histogramă pentru {selected_col}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown('<div class="sub-header">Box Plot</div>', unsafe_allow_html=True)
            fig_box = px.box(
                col_data,
                title=f"Box Plot pentru {selected_col}"
            )
            st.plotly_chart(fig_box, use_container_width=True)


# Cerința 4
def categorical_values():
    st.markdown('<h1 class="main-header">Analiza Coloanelor Categorice</h1>', unsafe_allow_html=True)

    with st.expander("Cerințe abordate în această secțiune", expanded=False):
        st.markdown("""
        • Identificare automată coloane categorice\n
        • Selectare coloană categorică de către utilizator\n
        • Creare count plot (bar chart) cu frecvențe\n
        • Afișare tabel cu frecvențe absolute și procente\n
        """)

    if 'df' not in st.session_state:
        st.warning("Te rog să încarci mai întâi datele din secțiunea 'Încărcarea Datelor'!")
        return

    df = st.session_state['df'].copy()

    st.markdown('<div class="sub-header">Analiza coloanelor categorice</div>', unsafe_allow_html=True)

    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if not categorical_cols:
        st.warning("Datasetul nu conține coloane categorice.")
    else:
        selected_cat_col = st.selectbox(
            "Selectează o coloană categorică:",
            categorical_cols
        )

        cat_data = df[selected_cat_col].dropna()

        if cat_data.empty:
            st.warning("Coloana selectată nu conține valori valide.")
        else:
            freq_abs = cat_data.value_counts()
            freq_pct = cat_data.value_counts(normalize=True) * 100

            freq_df = pd.DataFrame({
                "Categorie": freq_abs.index.astype(str),
                "Frecvență": freq_abs.values,
                "Procent": freq_pct.values.round(2)
            })

            fig = px.bar(
                freq_df,
                x="Categorie",
                y="Frecvență",
                text="Frecvență",
                title=f"Distribuția valorilor pentru {selected_cat_col}"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="sub-header">Tabel frecvențe</div>', unsafe_allow_html=True)
            st.dataframe(freq_df, use_container_width=True)


# Cerința 5
def corr_outlier():
    st.markdown('<h1 class="main-header">Corelații și Outlieri</h1>', unsafe_allow_html=True)

    with st.expander("Cerințe abordate în această secțiune", expanded=False):
        st.markdown("""
        • Calcul matrice de corelație pentru coloane numerice\n
        • Creare heatmap pentru vizualizarea corelațiilor\n
        • Selectare două variabile numerice\n
        • Creare scatter plot pentru cele două variabile\n
        • Calcul și afișare coeficient de corelație Pearson\n
        • Utilizare metoda IQR pentru detecție outlieri\n
        • Afișare număr și procent outlieri pentru fiecare coloană numerică\n
        • Vizualizare outlieri pe grafic\n
        """)

    if 'df' not in st.session_state:
        st.warning("Te rog să încarci mai întâi datele din secțiunea 'Încărcarea Datelor'!")
        return

    df = st.session_state['df'].copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("Nu există coloane numerice în dataset!")
        return

    st.markdown('<div class="sub-header">Matrice de corelație</div>', unsafe_allow_html=True)

    # Select correlation method
    corr_method = st.radio(
        "Metoda de corelație:",
        ['pearson', 'spearman', 'kendall'],
        format_func=lambda x: {
            'pearson': 'Pearson (Linear)',
            'spearman': 'Spearman (Rank)',
            'kendall': 'Kendall (Rank)'
        }[x],
        horizontal=True
    )

    # Calculate correlation
    corr_matrix = df[numeric_cols].corr(method=corr_method)

    # Heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='PRGn',
        color_continuous_midpoint=0,
        title=f'Heatmap Corelație ({corr_method.capitalize()})'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Strong correlations
    st.markdown("### Corelații Puternice (|r| > 0.7)")

    # Get upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_upper = corr_matrix.where(mask)

    # Flatten and filter
    strong_corr = []
    for col in corr_upper.columns:
        for idx in corr_upper.index:
            val = corr_upper.loc[idx, col]
            if not pd.isna(val) and abs(val) > 0.7:
                strong_corr.append({
                    'Variabila 1': idx,
                    'Variabila 2': col,
                    'Corelație': val,
                    'Forță': 'Foarte Puternică' if abs(val) > 0.9 else 'Puternică'
                })

    if strong_corr:
        strong_corr_df = pd.DataFrame(strong_corr).sort_values('Corelație', key=abs, ascending=False)
        st.dataframe(strong_corr_df, use_container_width=True)

        st.warning("""
             **Multicoliniaritate Potențială!**
        """)
    else:
        st.success("Nu există corelații foarte puternice (|r| > 0.7)")

    st.markdown('<div class="sub-header">Analiza relației dintre două variabile numerice</div>', unsafe_allow_html=True)

    if len(numeric_cols) < 2:
        st.warning("Sunt necesare cel puțin două coloane numerice.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            x_var = st.selectbox(
                "Selectează prima variabilă numerică (X):",
                numeric_cols,
                key="scatter_x"
            )

        with col2:
            y_var = st.selectbox(
                "Selectează a doua variabilă numerică (Y):",
                numeric_cols,
                index=1 if len(numeric_cols) > 1 else 0,
                key="scatter_y"
            )

        # Eliminare valori lipsă
        df_scatter = df[[x_var, y_var]].dropna()

        if len(df_scatter) < 2:
            st.error("Nu există suficiente date valide pentru a calcula corelația.")
        else:
            # Scatter plot
            fig = px.scatter(
                df_scatter,
                x=x_var,
                y=y_var,
                trendline="ols",
                title=f"Scatter plot: {x_var} vs {y_var}",
                labels={x_var: x_var, y_var: y_var}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Coeficient Pearson
            pearson_corr = df_scatter[x_var].corr(df_scatter[y_var], method='pearson')

            st.markdown(f"""
            <div class="success-box">
            <b>Coeficient de corelație Pearson:</b> <br>
            <h2 style="margin:0">{pearson_corr:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)

    # Metoda IQR și vizualizare outlieri
    st.markdown('<div class="sub-header">Metoda IQR și vizualizare outlieri</div>', unsafe_allow_html=True)

    col_for_box = st.selectbox("Selectează coloana pentru box plot:", numeric_cols, key="box_col")

    # Calculate IQR and outliers
    Q1 = df[col_for_box].quantile(0.25)
    Q3 = df[col_for_box].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outliers = df[(df[col_for_box] < lower_fence) | (df[col_for_box] > upper_fence)]
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / len(df) * 100) if len(df) > 0 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(" Total Valori", len(df))

    with col2:
        st.metric("Outlieri Găsiți", n_outliers)

    with col3:
        st.metric(" Procent Outlieri", f"{pct_outliers:.2f}%")

    # Box plot
    fig = px.box(
        df,
        y=col_for_box,
        points='outliers',
        title=f'Box Plot: {col_for_box}'
    )
    fig.add_hline(y=lower_fence, line_dash="dash", line_color="red", annotation_text="Lower Fence")
    fig.add_hline(y=upper_fence, line_dash="dash", line_color="red", annotation_text="Upper Fence")
    st.plotly_chart(fig, use_container_width=True)

    if n_outliers > 0:
        with st.expander(" Vezi Outlierii"):
            st.dataframe(outliers[[col_for_box]].describe(), use_container_width=True)
            st.dataframe(outliers.head(20), use_container_width=True)


if __name__ == "__main__":
    selected_module = sidebar_navigation()

    if selected_module == " Încărcarea Datelor":
        upload_files()
    if selected_module == " Filtrarea Datelor":
        filter_dataset()
    if selected_module == " Explorarea Datelor":
        dataset_info()
    if selected_module == " Analiza Coloanelor Numerice":
        numeric_values()
    if selected_module == " Analiza Coloanelor Categorice":
        categorical_values()
    elif selected_module == " Corelații și Outlieri":
        corr_outlier()
