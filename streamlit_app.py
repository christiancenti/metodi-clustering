import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import kmedoids
import warnings
warnings.filterwarnings('ignore')

# Configurazione della pagina
st.set_page_config(layout="wide", page_title="Clustering Methods Explainer")

st.title("Spiegazione Didattica dei Metodi di Clustering Non Supervisionato")

st.sidebar.header("Seleziona un Metodo")
method_options = [
    "Home",
    "Dataset Generator",
    "Clustering Gerarchico",
    "K-Means",
    "K-Medoids",
    "DBSCAN",
    "Confronto Metodi"
]
selected_method = st.sidebar.selectbox("Scegli un metodo da esplorare:", method_options)

# Funzione per generare dati
def generate_clustering_data(data_type="blobs", n_samples=300, n_features=2, n_centers=3, random_state=42):
    """Genera diversi tipi di dataset per clustering"""
    if data_type == "blobs":
        X, y_true = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features, 
                              random_state=random_state, cluster_std=1.0)
    elif data_type == "circles":
        X, y_true = make_circles(n_samples=n_samples, noise=0.1, factor=0.6, random_state=random_state)
    elif data_type == "moons":
        X, y_true = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    elif data_type == "anisotropic":
        X, y_true = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features,
                              random_state=random_state)
        # Trasformazione anisotropica
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    else:
        raise ValueError("data_type deve essere 'blobs', 'circles', 'moons', o 'anisotropic'")
    
    # Standardizza le features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y_true

# Funzione per addestrare automaticamente tutti i metodi di clustering
def addestra_tutti_clustering(X, y_true):
    """Addestra tutti i metodi di clustering con configurazioni standard ottimali"""
    
    st.session_state.clustering_results = {}
    
    with st.spinner("Addestramento di tutti i metodi di clustering in corso..."):
        # 1. Clustering Gerarchico - Ward con 3 cluster
        try:
            hierarchical = AgglomerativeClustering(n_clusters=3, linkage="ward")
            labels_hierarchical = hierarchical.fit_predict(X)
            metriche_hier = calcola_metriche_clustering(X, labels_hierarchical, y_true)
            
            st.session_state.clustering_results["Hierarchical"] = {
                "labels": labels_hierarchical,
                "metriche": metriche_hier,
                "parametri": {"linkage": "ward", "n_clusters": 3}
            }
        except Exception as e:
            st.warning(f"Errore nel Clustering Gerarchico: {str(e)}")
        
        # 2. K-Means - k=3, k-means++, 300 iterazioni
        try:
            kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, random_state=42)
            labels_kmeans = kmeans.fit_predict(X)
            metriche_kmeans = calcola_metriche_clustering(X, labels_kmeans, y_true)
            
            st.session_state.clustering_results["K-Means"] = {
                "labels": labels_kmeans,
                "metriche": metriche_kmeans,
                "parametri": {"k": 3, "init": "k-means++"},
                "centroids": kmeans.cluster_centers_,
                "inertia": kmeans.inertia_,
                "n_iter": kmeans.n_iter_
            }
        except Exception as e:
            st.warning(f"Errore in K-Means: {str(e)}")
        
        # 3. K-Medoids - FasterPAM, k=3, euclidean
        try:
            from sklearn.metrics.pairwise import euclidean_distances
            distance_matrix = euclidean_distances(X)
            result_kmedoids = kmedoids.fasterpam(distance_matrix, 3)
            
            labels_kmedoids = result_kmedoids.labels
            medoid_indices = result_kmedoids.medoids
            medoids_coords = X[medoid_indices]
            metriche_kmedoids = calcola_metriche_clustering(X, labels_kmedoids, y_true)
            
            st.session_state.clustering_results["K-Medoids"] = {
                "labels": labels_kmedoids,
                "metriche": metriche_kmedoids,
                "parametri": {"k": 3, "algorithm": "fasterpam", "metric": "euclidean"},
                "medoids": medoids_coords,
                "medoid_indices": medoid_indices,
                "loss": result_kmedoids.loss
            }
        except Exception as e:
            st.warning(f"Errore in K-Medoids: {str(e)}")
        
        # 4. DBSCAN - eps=0.5, min_samples=5
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels_dbscan = dbscan.fit_predict(X)
            
            n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
            n_noise = list(labels_dbscan).count(-1)
            
            # Calcola metriche escludendo noise points
            if n_clusters_dbscan > 1:
                mask = labels_dbscan != -1
                if np.sum(mask) > 0:
                    metriche_dbscan = calcola_metriche_clustering(X[mask], labels_dbscan[mask], 
                                                                y_true[mask] if y_true is not None else None)
                else:
                    metriche_dbscan = {"silhouette": -1, "calinski_harabasz": 0}
            else:
                metriche_dbscan = {"silhouette": -1, "calinski_harabasz": 0}
            
            st.session_state.clustering_results["DBSCAN"] = {
                "labels": labels_dbscan,
                "metriche": metriche_dbscan,
                "parametri": {"eps": 0.5, "min_samples": 5},
                "n_clusters": n_clusters_dbscan,
                "n_noise": n_noise
            }
        except Exception as e:
            st.warning(f"Errore in DBSCAN: {str(e)}")
    
    return len(st.session_state.clustering_results)

# Funzione per calcolare metriche di clustering
def calcola_metriche_clustering(X, labels_pred, labels_true=None):
    """Calcola metriche di valutazione per clustering"""
    metriche = {}
    
    # Silhouette Score (non richiede labels veri)
    if len(np.unique(labels_pred)) > 1:
        metriche["silhouette"] = silhouette_score(X, labels_pred)
        metriche["calinski_harabasz"] = calinski_harabasz_score(X, labels_pred)
    else:
        metriche["silhouette"] = -1
        metriche["calinski_harabasz"] = 0
    
    # Adjusted Rand Index (richiede labels veri)
    if labels_true is not None:
        metriche["adjusted_rand"] = adjusted_rand_score(labels_true, labels_pred)
    
    return metriche

# --- Pagina Home ---
if selected_method == "Home":
    st.header("Benvenuto!")
    st.markdown("""
    Questa applicazione è progettata per fornire una spiegazione didattica e interattiva
    dei principali metodi di **Clustering Non Supervisionato**.

    Il clustering è una tecnica di machine learning non supervisionato che raggruppa i dati
    in cluster (gruppi) basandosi sulla similarità tra i punti dati, senza conoscere
    le etichette vere dei gruppi.

    **Metodi disponibili:**
    1. **Clustering Gerarchico** - Costruisce una gerarchia di cluster
    2. **K-Means** - Partiziona i dati in k cluster usando centroidi
    3. **K-Medoids** - Simile a K-Means ma usa medoidi invece di centroidi
    4. **DBSCAN** - Clustering basato sulla densità, può trovare cluster di forma arbitraria

    **Procedura di utilizzo:**
    1. Vai alla sezione "Dataset Generator" e crea un dataset
    2. Esplora i diversi metodi di clustering
    3. Confronta i risultati nella sezione "Confronto Metodi"
    
    Utilizza il menu a sinistra per navigare tra le diverse sezioni.
    """)
    
    if "clustering_dataset" in st.session_state:
        st.success("Hai già generato un dataset. Puoi procedere con l'esplorazione dei metodi di clustering!")
    else:
        st.info("Per iniziare, vai alla sezione 'Dataset Generator' per creare un dataset.")

# --- Dataset Generator ---
elif selected_method == "Dataset Generator":
    st.header("Generazione del Dataset per Clustering")
    st.markdown("""
    In questa sezione puoi generare diversi tipi di dataset che saranno utilizzati
    per esplorare i vari metodi di clustering.
    """)
    
    # Parametri del dataset
    data_type = st.selectbox("Tipo di dataset:", 
                            ["blobs", "circles", "moons", "anisotropic"], 
                            key="clustering_data_type")
    
    # Descrizioni dei tipi di dataset
    descriptions = {
        "blobs": "Cluster sferici ben separati - ideale per K-Means",
        "circles": "Due cerchi concentrici - sfida per metodi basati su centroidi",
        "moons": "Due mezzelune intrecciate - forma non convessa",
        "anisotropic": "Cluster allungati con orientamenti diversi"
    }
    st.caption(descriptions[data_type])
    
    n_samples = st.slider("Numero di campioni:", 100, 1000, 300, key="clustering_samples")
    
    if data_type in ["blobs", "anisotropic"]:
        n_centers = st.slider("Numero di cluster veri:", 2, 8, 3, key="clustering_centers")
    else:
        n_centers = 2  # Fisso per circles e moons
    
    random_seed = st.slider("Seed casuale:", 1, 100, 42, key="clustering_seed")
    
    if st.button("Genera Dataset"):
        X, y_true = generate_clustering_data(data_type=data_type, n_samples=n_samples, 
                                           n_centers=n_centers, random_state=random_seed)
        
        # Salva il dataset nella session state
        st.session_state.clustering_dataset = {
            "X": X,
            "y_true": y_true,
            "data_type": data_type,
            "n_samples": n_samples,
            "n_centers": n_centers,
            "random_seed": random_seed
        }
        
        # Visualizzazione del dataset
        df_viz = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        df_viz["Cluster Vero"] = y_true
        
        fig = px.scatter(df_viz, x="Feature 1", y="Feature 2", color="Cluster Vero",
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        title=f"Dataset Generato: {data_type.title()}")
        st.plotly_chart(fig)
        
        # Addestra automaticamente tutti i metodi di clustering
        n_metodi = addestra_tutti_clustering(X, y_true)
        
        st.success(f"Dataset generato con successo! Sono stati addestrati automaticamente {n_metodi} metodi di clustering con configurazioni standard. Puoi visualizzare i risultati nelle sezioni specifiche o nel 'Confronto Metodi'.")
        
        # Mostra anteprima dei risultati
        if n_metodi > 0:
            st.subheader("Anteprima Risultati")
            
            # Crea una tabella riassuntiva
            preview_data = {
                "Metodo": [],
                "Silhouette Score": [],
                "N. Cluster": [],
                "Note": []
            }
            
            for method_name, results in st.session_state.clustering_results.items():
                preview_data["Metodo"].append(method_name)
                preview_data["Silhouette Score"].append(f"{results['metriche']['silhouette']:.3f}")
                
                if method_name == "DBSCAN":
                    preview_data["N. Cluster"].append(f"{results['n_clusters']} (+{results['n_noise']} noise)")
                    if results['n_clusters'] == 0:
                        preview_data["Note"].append("Nessun cluster trovato")
                    elif results['n_noise'] > len(X) * 0.3:
                        preview_data["Note"].append("Molti outliers")
                    else:
                        preview_data["Note"].append("✓ Outliers identificati")
                else:
                    n_clusters_found = len(np.unique(results["labels"]))
                    preview_data["N. Cluster"].append(str(n_clusters_found))
                    
                    if results['metriche']['silhouette'] > 0.5:
                        preview_data["Note"].append("✓ Buona separazione")
                    elif results['metriche']['silhouette'] > 0.3:
                        preview_data["Note"].append("⚖️ Separazione moderata")
                    else:
                        preview_data["Note"].append("⚠️ Separazione debole")
            
            df_preview = pd.DataFrame(preview_data)
            st.dataframe(df_preview, use_container_width=True)
            
            # Suggerimenti basati sul tipo di dataset
            st.subheader("Suggerimenti per questo Dataset")
            if data_type == "blobs":
                st.info("🎯 **Dataset ideale per:** K-Means e K-Medoids (cluster sferici ben separati)")
                st.info("📊 **Aspettative:** Tutti i metodi dovrebbero performare bene, con K-Means leggermente favorito")
            elif data_type == "circles":
                st.info("🎯 **Dataset ideale per:** DBSCAN (forme circolari)")
                st.warning("⚠️ **Sfida per:** K-Means e K-Medoids (assumono cluster sferici)")
                st.info("📊 **Aspettative:** DBSCAN dovrebbe superare significativamente gli altri metodi")
            elif data_type == "moons":
                st.info("🎯 **Dataset ideale per:** DBSCAN (forme curve)")
                st.warning("⚠️ **Sfida per:** K-Means, K-Medoids e Clustering Gerarchico")
                st.info("📊 **Aspettative:** DBSCAN dovrebbe essere l'unico a catturare correttamente la struttura")
            elif data_type == "anisotropic":
                st.info("🎯 **Dataset moderatamente favorevole per:** K-Medoids e DBSCAN")
                st.warning("⚠️ **Sfida per:** K-Means (cluster allungati)")
                st.info("📊 **Aspettative:** K-Medoids potrebbe essere più robusto di K-Means")
            
            # Identifica il metodo migliore
            best_method = max(st.session_state.clustering_results.items(), 
                            key=lambda x: x[1]['metriche']['silhouette'])
            st.success(f"🏆 **Miglior metodo per questo dataset:** {best_method[0]} (Silhouette: {best_method[1]['metriche']['silhouette']:.3f})")
        else:
            st.error("Nessun metodo di clustering è stato addestrato con successo. Verifica la configurazione del dataset.")

# --- Clustering Gerarchico ---
elif selected_method == "Clustering Gerarchico":
    st.header("Clustering Gerarchico")
    st.markdown("""
    Il **Clustering Gerarchico** costruisce una gerarchia di cluster che può essere rappresentata
    come un albero (dendrogramma). Implementiamo l'approccio **agglomerativo (bottom-up)**.
    
    **Configurazione utilizzata:**
    - **Criterio di legame:** Ward (minimizza la varianza intra-cluster)
    - **Numero di cluster:** 3 (valore standard bilanciato)
    - **Metrica di distanza:** Euclidea (richiesta per Ward)
    
    **Criteri di Legame disponibili:**
    - **Ward:** Minimizza la varianza intra-cluster (ottimale per cluster sferici)
    - **Complete:** Distanza massima tra punti di cluster diversi
    - **Average:** Distanza media tra tutti i punti di cluster diversi
    - **Single:** Distanza minima tra punti di cluster diversi
    
    **Vantaggi:**
    - Non richiede di specificare il numero di cluster a priori (dal dendrogramma)
    - Produce una gerarchia completa di soluzioni
    - Deterministico (sempre lo stesso risultato)
    - Ward è particolarmente efficace per cluster compatti
    
    **Svantaggi:**
    - Complessità computazionale O(n³)
    - Sensibile a outliers
    - Decisioni di merge sono definitive (non reversibili)
    """)

    if "clustering_dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        X = st.session_state.clustering_dataset["X"]
        y_true = st.session_state.clustering_dataset["y_true"]
        
        # Configurazioni standard ottimali
        linkage_type = "ward"  # Criterio più efficace per cluster compatti
        n_clusters = 3  # Valore standard bilanciato
        
        st.info(f"Configurazione utilizzata: Linkage={linkage_type}, n_clusters={n_clusters}")
        
        if st.button("Esegui Clustering Gerarchico"):
            # Clustering gerarchico con configurazione standard
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, 
                                                 linkage=linkage_type)
            labels_hierarchical = hierarchical.fit_predict(X)
            
            # Calcola metriche
            metriche = calcola_metriche_clustering(X, labels_hierarchical, y_true)
            
            # Salva risultati
            st.session_state.clustering_results["Hierarchical"] = {
                "labels": labels_hierarchical,
                "metriche": metriche,
                "parametri": {"linkage": linkage_type, "n_clusters": n_clusters}
            }
            
            # Visualizzazione risultati
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risultati Clustering")
                df_result = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
                df_result["Cluster"] = labels_hierarchical
                
                fig_result = px.scatter(df_result, x="Feature 1", y="Feature 2", 
                                      color="Cluster", 
                                      color_discrete_sequence=px.colors.qualitative.Set1,
                                      title=f"Clustering Gerarchico (Ward)")
                st.plotly_chart(fig_result)
                
                # Metriche
                st.write("**Metriche di Valutazione:**")
                st.write(f"- Silhouette Score: {metriche['silhouette']:.3f}")
                st.write(f"- Calinski-Harabasz Index: {metriche['calinski_harabasz']:.3f}")
                if 'adjusted_rand' in metriche:
                    st.write(f"- Adjusted Rand Index: {metriche['adjusted_rand']:.3f}")
            
            with col2:
                st.subheader("Dendrogramma")
                
                # Calcola linkage matrix per dendrogramma
                linkage_matrix = linkage(X, method=linkage_type)
                
                # Crea dendrogramma
                fig_dend, ax = plt.subplots(figsize=(10, 6))
                dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=5)
                ax.set_title(f"Dendrogramma (Ward)")
                ax.set_xlabel("Indice Campione o (Dimensione Cluster)")
                ax.set_ylabel("Distanza")
                st.pyplot(fig_dend)
                
                st.markdown("""
                **Come leggere il dendrogramma:**
                - L'asse Y mostra la distanza a cui i cluster vengono uniti
                - Tagliando orizzontalmente a diverse altezze si ottengono diversi numeri di cluster
                - Rami più lunghi indicano cluster più distinti
                - Ward minimizza la varianza, creando cluster compatti
                """)
        
        # Sezione educativa sui diversi criteri di legame
        st.subheader("Confronto dei Criteri di Legame")
        if st.button("Confronta tutti i criteri di legame"):
            linkage_methods = ["ward", "complete", "average", "single"]
            
            cols = st.columns(2)
            for i, method in enumerate(linkage_methods):
                with cols[i % 2]:
                    # Esegui clustering con il metodo corrente
                    hierarchical_temp = AgglomerativeClustering(n_clusters=n_clusters, 
                                                              linkage=method)
                    labels_temp = hierarchical_temp.fit_predict(X)
                    metriche_temp = calcola_metriche_clustering(X, labels_temp, y_true)
                    
                    # Visualizza risultato
                    df_temp = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
                    df_temp["Cluster"] = labels_temp
                    
                    fig_temp = px.scatter(df_temp, x="Feature 1", y="Feature 2", 
                                        color="Cluster",
                                        color_discrete_sequence=px.colors.qualitative.Set1,
                                        title=f"{method.title()} (Sil: {metriche_temp['silhouette']:.3f})")
                    fig_temp.update_layout(height=400)
                    st.plotly_chart(fig_temp, use_container_width=True)

# --- K-Means ---
elif selected_method == "K-Means":
    st.header("K-Means Clustering")
    st.markdown("""
    **K-Means** è uno degli algoritmi di clustering più popolari. Partiziona i dati in k cluster
    minimizzando la somma delle distanze quadratiche dai punti ai centroidi dei cluster.
    
    **Configurazione utilizzata:**
    - **Numero di cluster (k):** 3 (valore standard bilanciato)
    - **Inizializzazione:** k-means++ (migliore convergenza rispetto a random)
    - **Numero massimo iterazioni:** 300 (default ottimale)
    - **Algoritmo:** Lloyd (default di scikit-learn)
    
    **Come funziona:**
    1. Scegli il numero di cluster k
    2. Inizializza k centroidi con k-means++ (smart initialization)
    3. Assegna ogni punto al centroide più vicino
    4. Ricalcola i centroidi come media dei punti assegnati
    5. Ripeti i passi 3-4 fino alla convergenza
    
    **Vantaggi:**
    - Semplice e veloce (complessità lineare O(n))
    - Funziona bene con cluster sferici e ben separati
    - k-means++ garantisce buona inizializzazione
    - Scalabile a grandi dataset
    
    **Svantaggi:**
    - Richiede di specificare k a priori
    - Assume cluster sferici di dimensioni simili
    - Sensibile a outliers
    - Può convergere a minimi locali
    
    **Metodo del Gomito:** Tecnica per scegliere k ottimale analizzando la diminuzione dell'inerzia.
    """)

    if "clustering_dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        X = st.session_state.clustering_dataset["X"]
        y_true = st.session_state.clustering_dataset["y_true"]
        
        # Configurazioni standard ottimali
        k_clusters = 3  # Valore standard bilanciato
        init_method = "k-means++"  # Inizializzazione intelligente
        max_iter = 300  # Default ottimale
        
        st.info(f"Configurazione utilizzata: k={k_clusters}, init={init_method}, max_iter={max_iter}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Esegui K-Means"):
                # K-Means clustering con configurazione standard
                kmeans = KMeans(n_clusters=k_clusters, init=init_method, 
                               max_iter=max_iter, random_state=42)
                labels_kmeans = kmeans.fit_predict(X)
                
                # Calcola metriche
                metriche = calcola_metriche_clustering(X, labels_kmeans, y_true)
                
                # Salva risultati
                st.session_state.clustering_results["K-Means"] = {
                    "labels": labels_kmeans,
                    "metriche": metriche,
                    "parametri": {"k": k_clusters, "init": init_method},
                    "centroids": kmeans.cluster_centers_,
                    "inertia": kmeans.inertia_,
                    "n_iter": kmeans.n_iter_
                }
                
                # Visualizzazione risultati
                df_result = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
                df_result["Cluster"] = labels_kmeans
                
                fig_result = px.scatter(df_result, x="Feature 1", y="Feature 2", 
                                      color="Cluster",
                                      color_discrete_sequence=px.colors.qualitative.Set1,
                                      title=f"K-Means (k={k_clusters})")
                
                # Aggiungi centroidi
                centroids_df = pd.DataFrame(kmeans.cluster_centers_, 
                                          columns=["Feature 1", "Feature 2"])
                fig_result.add_trace(go.Scatter(x=centroids_df["Feature 1"], 
                                              y=centroids_df["Feature 2"],
                                              mode='markers',
                                              marker=dict(symbol='x', size=15, color='black'),
                                              name='Centroidi'))
                
                st.plotly_chart(fig_result)
                
                # Metriche e informazioni
                st.write("**Risultati K-Means:**")
                st.write(f"- Inerzia (WCSS): {kmeans.inertia_:.3f}")
                st.write(f"- Iterazioni per convergenza: {kmeans.n_iter_}")
                st.write(f"- Silhouette Score: {metriche['silhouette']:.3f}")
                st.write(f"- Calinski-Harabasz Index: {metriche['calinski_harabasz']:.3f}")
                if 'adjusted_rand' in metriche:
                    st.write(f"- Adjusted Rand Index: {metriche['adjusted_rand']:.3f}")
        
        with col2:
            if st.button("Analisi Metodo del Gomito"):
                st.subheader("Metodo del Gomito")
                
                # Calcola inerzia per diversi valori di k
                k_range = range(1, 11)
                inertias = []
                silhouette_scores = []
                
                with st.spinner("Calcolo curve di valutazione..."):
                    for k in k_range:
                        kmeans_temp = KMeans(n_clusters=k, init=init_method, random_state=42)
                        kmeans_temp.fit(X)
                        inertias.append(kmeans_temp.inertia_)
                        
                        # Silhouette score solo per k > 1
                        if k > 1:
                            sil_score = silhouette_score(X, kmeans_temp.labels_)
                            silhouette_scores.append(sil_score)
                        else:
                            silhouette_scores.append(0)
                
                # Grafico del gomito
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias,
                                             mode='lines+markers',
                                             name='Inerzia (WCSS)',
                                             line=dict(color='blue')))
                fig_elbow.update_layout(title="Metodo del Gomito",
                                      xaxis_title="Numero di Cluster (k)",
                                      yaxis_title="Inerzia (WCSS)")
                st.plotly_chart(fig_elbow)
                
                # Grafico Silhouette Score
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(x=list(k_range)[1:], y=silhouette_scores[1:],
                                           mode='lines+markers',
                                           name='Silhouette Score',
                                           line=dict(color='red')))
                fig_sil.update_layout(title="Silhouette Score vs k",
                                    xaxis_title="Numero di Cluster (k)",
                                    yaxis_title="Silhouette Score")
                st.plotly_chart(fig_sil)
                
                # Suggerimenti per k ottimale
                best_k_silhouette = np.argmax(silhouette_scores[1:]) + 2  # +2 perché iniziamo da k=2
                st.write("**Suggerimenti per k ottimale:**")
                st.write(f"- Metodo del Gomito: Cerca il 'gomito' nella curva dell'inerzia")
                st.write(f"- Silhouette Score: k={best_k_silhouette} ha il punteggio più alto ({max(silhouette_scores[1:]):.3f})")
                
                st.markdown("""
                **Come interpretare:**
                - **Metodo del Gomito:** Cerca il punto dove la diminuzione dell'inerzia rallenta
                - **Silhouette Score:** Valori più alti indicano cluster meglio separati
                - **Bilanciamento:** Considera sia la qualità dei cluster che la semplicità del modello
                """)
        
        # Sezione educativa sui limiti di K-Means
        st.subheader("Limiti di K-Means")
        if st.button("Mostra esempi di fallimento di K-Means"):
            st.markdown("""
            **K-Means funziona meglio con:**
            - Cluster sferici e compatti
            - Cluster di dimensioni simili
            - Cluster ben separati
            
            **K-Means ha difficoltà con:**
            - Cluster di forme non sferiche (es. moons, circles)
            - Cluster di densità molto diverse
            - Presenza di molti outliers
            """)
            
            # Mostra performance su diversi tipi di dataset se disponibili
            if st.session_state.clustering_dataset["data_type"] in ["circles", "moons"]:
                st.warning(f"Il dataset corrente ({st.session_state.clustering_dataset['data_type']}) presenta sfide per K-Means a causa della forma non sferica dei cluster.")
            elif st.session_state.clustering_dataset["data_type"] == "anisotropic":
                st.warning("Il dataset corrente (anisotropic) presenta cluster allungati che possono essere difficili per K-Means.")
            else:
                st.success("Il dataset corrente (blobs) è ideale per K-Means con cluster sferici ben separati.")

# --- K-Medoids ---
elif selected_method == "K-Medoids":
    st.header("K-Medoids Clustering")
    st.markdown("""
    **K-Medoids** (anche noto come PAM - Partitioning Around Medoids) è simile a K-Means
    ma usa **medoidi** invece di centroidi.
    
    **Configurazione utilizzata:**
    - **Numero di cluster (k):** 3 (valore standard bilanciato)
    - **Algoritmo:** FasterPAM (ottimizzato con complessità O(k) per iterazione)
    - **Metrica di distanza:** Euclidea (standard per confronto con K-Means)
    
    **Differenze chiave con K-Means:**
    - **Medoide:** Un punto dati reale che rappresenta il cluster (non la media)
    - **Centroide:** Punto medio calcolato (può non essere un punto dati reale)
    
    **Come funziona:**
    1. Scegli k medoidi iniziali casualmente
    2. Assegna ogni punto al medoide più vicino
    3. Per ogni cluster, trova il punto che minimizza la distanza totale agli altri punti del cluster
    4. Se questo punto è diverso dal medoide attuale, sostituiscilo
    5. Ripeti fino alla convergenza
    
    **Vantaggi rispetto a K-Means:**
    - Più robusto agli outliers
    - Funziona con qualsiasi metrica di distanza
    - I medoidi sono punti dati reali (più interpretabili)
    - Meno sensibile all'inizializzazione
    
    **Svantaggi:**
    - Più lento di K-Means (complessità O(n²) per PAM classico)
    - Ancora richiede di specificare k a priori
    - FasterPAM riduce la complessità ma rimane più costoso di K-Means
    
    **Algoritmi disponibili:**
    - **FasterPAM:** Versione ottimizzata con complessità O(k) per iterazione (utilizzato)
    - **PAM:** Algoritmo classico, più lento ma stabile
    - **FastPAM1:** Variante veloce di PAM
    """)

    if "clustering_dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        X = st.session_state.clustering_dataset["X"]
        y_true = st.session_state.clustering_dataset["y_true"]
        
        # Configurazioni standard ottimali
        k_clusters = 3  # Valore standard bilanciato
        algorithm = "fasterpam"  # Algoritmo più efficiente
        metric = "euclidean"  # Standard per confronto con K-Means
        
        st.info(f"Configurazione utilizzata: k={k_clusters}, algoritmo={algorithm.upper()}, metrica={metric}")
        
        if st.button("Esegui K-Medoids"):
            from sklearn.metrics.pairwise import euclidean_distances
            
            with st.spinner(f"Esecuzione {algorithm.upper()}..."):
                # Calcola matrice delle distanze
                distance_matrix = euclidean_distances(X)
                
                # Esegui K-Medoids con FasterPAM
                result = kmedoids.fasterpam(distance_matrix, k_clusters)
                
                labels_kmedoids = result.labels
                medoid_indices = result.medoids
                loss = result.loss
                
                # Ottieni le coordinate dei medoidi
                medoids_coords = X[medoid_indices]
                
                # Calcola metriche
                metriche = calcola_metriche_clustering(X, labels_kmedoids, y_true)
                
                # Salva risultati
                st.session_state.clustering_results["K-Medoids"] = {
                    "labels": labels_kmedoids,
                    "metriche": metriche,
                    "parametri": {"k": k_clusters, "algorithm": algorithm, "metric": metric},
                    "medoids": medoids_coords,
                    "medoid_indices": medoid_indices,
                    "loss": loss
                }
                
                # Visualizzazione risultati
                df_result = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
                df_result["Cluster"] = labels_kmedoids
                
                fig_result = px.scatter(df_result, x="Feature 1", y="Feature 2", 
                                      color="Cluster",
                                      color_discrete_sequence=px.colors.qualitative.Set1,
                                      title=f"K-Medoids (FasterPAM, k={k_clusters})")
                
                # Aggiungi medoidi
                medoids_df = pd.DataFrame(medoids_coords, 
                                        columns=["Feature 1", "Feature 2"])
                fig_result.add_trace(go.Scatter(x=medoids_df["Feature 1"], 
                                              y=medoids_df["Feature 2"],
                                              mode='markers',
                                              marker=dict(symbol='diamond', size=15, color='black'),
                                              name='Medoidi'))
                
                st.plotly_chart(fig_result)
                
                # Metriche
                st.write("**Risultati K-Medoids:**")
                st.write(f"- Loss (costo totale): {loss:.3f}")
                st.write(f"- Silhouette Score: {metriche['silhouette']:.3f}")
                st.write(f"- Calinski-Harabasz Index: {metriche['calinski_harabasz']:.3f}")
                if 'adjusted_rand' in metriche:
                    st.write(f"- Adjusted Rand Index: {metriche['adjusted_rand']:.3f}")
                
                # Informazioni sui medoidi
                st.subheader("Informazioni sui Medoidi")
                st.write("**Indici e coordinate dei medoidi nel dataset:**")
                for i, idx in enumerate(medoid_indices):
                    st.write(f"- Cluster {i}: Punto {idx} alle coordinate ({medoids_coords[i][0]:.3f}, {medoids_coords[i][1]:.3f})")
                
                # Confronto con K-Means se disponibile
                if "K-Means" in st.session_state.clustering_results:
                    st.subheader("Confronto con K-Means")
                    kmeans_results = st.session_state.clustering_results["K-Means"]
                    kmeans_silhouette = kmeans_results["metriche"]["silhouette"]
                    kmeans_inertia = kmeans_results["inertia"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**K-Means:**")
                        st.write(f"- Silhouette: {kmeans_silhouette:.3f}")
                        st.write(f"- Inerzia: {kmeans_inertia:.3f}")
                        st.write("- Centroidi: punti virtuali")
                    
                    with col2:
                        st.write("**K-Medoids:**")
                        st.write(f"- Silhouette: {metriche['silhouette']:.3f}")
                        st.write(f"- Loss: {loss:.3f}")
                        st.write("- Medoidi: punti reali del dataset")
                    
                    if metriche['silhouette'] > kmeans_silhouette:
                        st.success("✅ K-Medoids ha ottenuto un Silhouette Score migliore!")
                    elif abs(metriche['silhouette'] - kmeans_silhouette) < 0.01:
                        st.info("⚖️ K-Medoids e K-Means hanno performance simili.")
                    else:
                        st.info("📊 K-Means ha ottenuto un Silhouette Score migliore.")
                    
                    st.markdown("""
                    **Differenze chiave osservate:**
                    - **Interpretabilità:** I medoidi sono punti dati reali, più facili da interpretare
                    - **Robustezza:** K-Medoids è meno sensibile agli outliers
                    - **Velocità:** K-Means è generalmente più veloce per dataset grandi
                    - **Flessibilità:** K-Medoids funziona con qualsiasi metrica di distanza
                    """)
        
        # Sezione educativa sui diversi algoritmi
        st.subheader("Confronto Algoritmi K-Medoids")
        if st.button("Confronta algoritmi PAM"):
            from sklearn.metrics.pairwise import euclidean_distances
            
            algorithms = ["fasterpam", "pam", "fastpam1"]
            algorithm_names = ["FasterPAM", "PAM", "FastPAM1"]
            
            cols = st.columns(3)
            
            with st.spinner("Confronto algoritmi in corso..."):
                for i, (alg, name) in enumerate(zip(algorithms, algorithm_names)):
                    with cols[i]:
                        try:
                            # Calcola matrice distanze
                            distance_matrix = euclidean_distances(X)
                            
                            # Esegui algoritmo
                            if alg == "fasterpam":
                                result = kmedoids.fasterpam(distance_matrix, k_clusters)
                            elif alg == "pam":
                                result = kmedoids.pam(distance_matrix, k_clusters)
                            elif alg == "fastpam1":
                                result = kmedoids.fastpam1(distance_matrix, k_clusters)
                            
                            # Calcola metriche
                            metriche_temp = calcola_metriche_clustering(X, result.labels, y_true)
                            
                            # Visualizza
                            df_temp = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
                            df_temp["Cluster"] = result.labels
                            
                            fig_temp = px.scatter(df_temp, x="Feature 1", y="Feature 2", 
                                                color="Cluster",
                                                color_discrete_sequence=px.colors.qualitative.Set1,
                                                title=f"{name}")
                            fig_temp.update_layout(height=400)
                            st.plotly_chart(fig_temp, use_container_width=True)
                            
                            st.write(f"**{name}:**")
                            st.write(f"- Silhouette: {metriche_temp['silhouette']:.3f}")
                            st.write(f"- Loss: {result.loss:.3f}")
                            
                        except Exception as e:
                            st.error(f"Errore con {name}: {str(e)}")
            
            st.markdown("""
            **Caratteristiche degli algoritmi:**
            - **FasterPAM:** Più veloce, complessità O(k) per iterazione
            - **PAM:** Algoritmo classico, più lento ma molto stabile
            - **FastPAM1:** Compromesso tra velocità e stabilità
            """)
        
        # Sezione educativa sui vantaggi dei medoidi
        st.subheader("Quando usare K-Medoids")
        st.markdown("""
        **K-Medoids è preferibile quando:**
        - I dati contengono outliers significativi
        - È importante che i rappresentanti dei cluster siano punti reali
        - Si lavora con metriche di distanza non euclidee
        - L'interpretabilità dei centri cluster è cruciale
        
        **K-Means è preferibile quando:**
        - I cluster sono approssimativamente sferici
        - La velocità di calcolo è prioritaria
        - Si lavora con dataset molto grandi
        - I centroidi virtuali sono accettabili
        """)
        
        # Mostra suggerimenti basati sul tipo di dataset
        if st.session_state.clustering_dataset["data_type"] in ["circles", "moons"]:
            st.info("💡 Per dataset con forme non sferiche come questo, considera DBSCAN che può gestire forme arbitrarie.")
        elif st.session_state.clustering_dataset["data_type"] == "anisotropic":
            st.info("💡 Per cluster allungati, K-Medoids può essere più robusto di K-Means.")
        else:
            st.info("💡 Per cluster sferici ben separati come questi, K-Means e K-Medoids dovrebbero avere performance simili.")

# --- DBSCAN ---
elif selected_method == "DBSCAN":
    st.header("DBSCAN Clustering")
    st.markdown("""
    **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) è un algoritmo
    di clustering basato sulla densità che può trovare cluster di forma arbitraria.
    
    **Configurazione utilizzata:**
    - **Eps (ε):** 0.5 (raggio del vicinato - valore standard bilanciato)
    - **MinPts:** 5 (punti minimi per cluster - regola empirica per dati 2D)
    - **Metrica:** Euclidea (default di scikit-learn)
    
    **Concetti chiave:**
    - **Eps (ε):** Raggio massimo del vicinato di un punto
    - **MinPts:** Numero minimo di punti richiesti per formare un cluster denso
    - **Core Point:** Punto con almeno MinPts punti nel suo ε-vicinato
    - **Border Point:** Punto nel ε-vicinato di un core point ma non core esso stesso
    - **Noise Point:** Punto che non è né core né border (outlier)
    
    **Come funziona:**
    1. Per ogni punto, conta i vicini entro distanza ε
    2. Se un punto ha ≥ MinPts vicini, è un core point
    3. Forma cluster connettendo core points vicini
    4. Aggiungi border points ai cluster dei core points vicini
    5. I punti rimanenti sono noise/outliers
    
    **Vantaggi:**
    - Non richiede di specificare il numero di cluster a priori
    - Trova cluster di forma arbitraria (non solo sferici)
    - Identifica automaticamente outliers
    - Robusto al rumore
    - Efficace per cluster di densità simile
    
    **Svantaggi:**
    - Sensibile ai parametri ε e MinPts
    - Difficoltà con cluster di densità molto diverse
    - Prestazioni degradano in alta dimensionalità
    - Può essere sensibile alla scala dei dati
    """)

    if "clustering_dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        X = st.session_state.clustering_dataset["X"]
        y_true = st.session_state.clustering_dataset["y_true"]
        
        # Configurazioni standard ottimali
        eps = 0.5  # Valore standard bilanciato
        min_samples = 5  # Regola empirica: dimensionalità + 1, poi arrotondata per robustezza
        
        st.info(f"Configurazione utilizzata: ε={eps}, MinPts={min_samples}")
        
        if st.button("Esegui DBSCAN"):
            # DBSCAN clustering con configurazione standard
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels_dbscan = dbscan.fit_predict(X)
            
            # Conta cluster e noise
            n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
            n_noise = list(labels_dbscan).count(-1)
            
            # Calcola metriche (escludendo noise points per silhouette)
            if n_clusters > 1:
                # Rimuovi noise points per calcolo metriche
                mask = labels_dbscan != -1
                if np.sum(mask) > 0:
                    metriche = calcola_metriche_clustering(X[mask], labels_dbscan[mask], 
                                                         y_true[mask] if y_true is not None else None)
                else:
                    metriche = {"silhouette": -1, "calinski_harabasz": 0}
            else:
                metriche = {"silhouette": -1, "calinski_harabasz": 0}
            
            # Salva risultati
            st.session_state.clustering_results["DBSCAN"] = {
                "labels": labels_dbscan,
                "metriche": metriche,
                "parametri": {"eps": eps, "min_samples": min_samples},
                "n_clusters": n_clusters,
                "n_noise": n_noise
            }
            
            # Visualizzazione risultati
            df_result = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
            df_result["Cluster"] = labels_dbscan
            
            # Sostituisci -1 (noise) con "Noise" per visualizzazione
            df_result["Cluster_Label"] = df_result["Cluster"].apply(
                lambda x: "Noise" if x == -1 else f"Cluster {x}"
            )
            
            fig_result = px.scatter(df_result, x="Feature 1", y="Feature 2", 
                                  color="Cluster_Label",
                                  color_discrete_sequence=px.colors.qualitative.Set1,
                                  title=f"DBSCAN (ε={eps}, MinPts={min_samples})")
            
            st.plotly_chart(fig_result)
            
            # Statistiche dettagliate
            st.write("**Risultati DBSCAN:**")
            st.write(f"- Numero di cluster trovati: {n_clusters}")
            st.write(f"- Numero di punti noise: {n_noise}")
            st.write(f"- Percentuale noise: {n_noise/len(X)*100:.1f}%")
            st.write(f"- Punti in cluster: {len(X) - n_noise}")
            
            # Analisi dei cluster trovati
            if n_clusters > 0:
                cluster_sizes = []
                for i in range(n_clusters):
                    size = np.sum(labels_dbscan == i)
                    cluster_sizes.append(size)
                
                st.write("**Dimensioni dei cluster:**")
                for i, size in enumerate(cluster_sizes):
                    st.write(f"- Cluster {i}: {size} punti ({size/len(X)*100:.1f}%)")
            
            # Metriche
            if n_clusters > 1:
                st.write("**Metriche di Valutazione:**")
                st.write(f"- Silhouette Score: {metriche['silhouette']:.3f}")
                st.write(f"- Calinski-Harabasz Index: {metriche['calinski_harabasz']:.3f}")
                if 'adjusted_rand' in metriche:
                    st.write(f"- Adjusted Rand Index: {metriche['adjusted_rand']:.3f}")
            else:
                st.warning("Non sono stati trovati cluster sufficienti per calcolare le metriche.")
                st.info("💡 Prova a diminuire ε o MinPts per trovare più cluster.")
            
            # Confronto con altri metodi se disponibili
            if len(st.session_state.clustering_results) > 1:
                st.subheader("Confronto con Altri Metodi")
                
                comparison_methods = []
                comparison_silhouettes = []
                
                for method_name, results in st.session_state.clustering_results.items():
                    if method_name != "DBSCAN" and "silhouette" in results["metriche"]:
                        comparison_methods.append(method_name)
                        comparison_silhouettes.append(results["metriche"]["silhouette"])
                
                if comparison_methods:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Altri metodi:**")
                        for method, sil in zip(comparison_methods, comparison_silhouettes):
                            st.write(f"- {method}: {sil:.3f}")
                    
                    with col2:
                        st.write("**DBSCAN:**")
                        if n_clusters > 1:
                            st.write(f"- Silhouette: {metriche['silhouette']:.3f}")
                            st.write(f"- Cluster trovati: {n_clusters}")
                            st.write(f"- Outliers identificati: {n_noise}")
                        else:
                            st.write("- Nessun cluster valido trovato")
                    
                    # Valutazione delle performance
                    if n_clusters > 1:
                        best_other_sil = max(comparison_silhouettes) if comparison_silhouettes else -1
                        if metriche['silhouette'] > best_other_sil:
                            st.success("✅ DBSCAN ha ottenuto il miglior Silhouette Score!")
                        elif abs(metriche['silhouette'] - best_other_sil) < 0.05:
                            st.info("⚖️ DBSCAN ha performance competitive.")
                        else:
                            st.info("📊 Altri metodi hanno ottenuto Silhouette Score migliori.")
                        
                        st.markdown("""
                        **Vantaggi unici di DBSCAN:**
                        - Identifica automaticamente outliers
                        - Non assume forme sferiche dei cluster
                        - Non richiede di specificare il numero di cluster
                        - Robusto a cluster di forme irregolari
                        """)
        
        # Sezione educativa sui parametri
        st.subheader("Comprensione dei Parametri")
        st.markdown(f"""
        **Parametri utilizzati (ε={eps}, MinPts={min_samples}):**
        
        **Eps (ε) = {eps}:**
        - Raggio del vicinato per ogni punto
        - Valore standard che bilancia sensibilità e robustezza
        - Troppo piccolo → molti punti diventano noise
        - Troppo grande → tutti i punti finiscono in un cluster
        
        **MinPts = {min_samples}:**
        - Numero minimo di punti per formare un cluster denso
        - Regola empirica: dimensionalità + 1 (per 2D: ≥ 3)
        - Valore {min_samples} fornisce buona robustezza al rumore
        - Valori più alti → cluster più densi richiesti
        """)
        
        # Suggerimenti basati sul tipo di dataset
        data_type = st.session_state.clustering_dataset["data_type"]
        
        st.subheader("Adattabilità al Dataset")
        if data_type == "circles":
            st.success("🎯 DBSCAN è ideale per questo dataset con cerchi concentrici!")
            st.info("DBSCAN può identificare la struttura circolare che K-Means non riesce a catturare.")
        elif data_type == "moons":
            st.success("🎯 DBSCAN è perfetto per questo dataset con mezzelune!")
            st.info("DBSCAN gestisce bene le forme curve e non convesse.")
        elif data_type == "anisotropic":
            st.info("🔧 DBSCAN può gestire cluster allungati meglio di K-Means.")
            st.info("La forma anisotropica non è un problema per DBSCAN.")
        else:  # blobs
            st.info("📊 Per cluster sferici come questi, DBSCAN compete con K-Means.")
            st.info("DBSCAN può comunque identificare outliers che K-Means non rileva.")
        
        # Sezione educativa avanzata
        st.subheader("Quando usare DBSCAN")
        if st.button("Mostra guida all'uso di DBSCAN"):
            st.markdown("""
            **DBSCAN è la scelta migliore quando:**
            - I cluster hanno forme irregolari o non sferiche
            - È importante identificare outliers/anomalie
            - Non si conosce a priori il numero di cluster
            - I cluster hanno densità simile ma forme diverse
            - Si lavora con dati spaziali o geografici
            
            **DBSCAN ha limitazioni quando:**
            - I cluster hanno densità molto diverse
            - I dati sono ad alta dimensionalità (>10-15 dimensioni)
            - È necessario assegnare ogni punto a un cluster (no outliers)
            - I cluster sono molto piccoli o sparsi
            
            **Confronto con altri metodi:**
            - **vs K-Means:** DBSCAN gestisce forme arbitrarie, K-Means è più veloce
            - **vs Hierarchical:** DBSCAN identifica outliers, Hierarchical fornisce gerarchia
            - **vs K-Medoids:** DBSCAN non richiede k, K-Medoids è più interpretabile
            """)
        
        # Suggerimenti per ottimizzazione parametri
        st.subheader("Ottimizzazione Parametri (Avanzato)")
        if st.button("Mostra tecniche per scegliere ε"):
            st.markdown("""
            **Metodo k-distance per scegliere ε:**
            1. Per ogni punto, calcola la distanza al k-esimo vicino più prossimo (k = MinPts)
            2. Ordina queste distanze in modo crescente
            3. Traccia il grafico delle distanze ordinate
            4. Cerca il "gomito" (punto di massima curvatura)
            5. Il valore di ε corrispondente al gomito è spesso ottimale
            
            **Regole empiriche per MinPts:**
            - Dati 2D: MinPts ≥ 3 (utilizzato: {min_samples})
            - Dati nD: MinPts ≥ n + 1
            - Per dati rumorosi: aumentare MinPts
            - Per cluster piccoli: diminuire MinPts
            
            **Validazione:**
            - Usa Silhouette Score per valutare la qualità
            - Controlla la percentuale di noise (5-20% è spesso accettabile)
            - Verifica che i cluster abbiano senso nel dominio applicativo
            """.format(min_samples=min_samples))

# --- Confronto Metodi ---
elif selected_method == "Confronto Metodi":
    st.header("Confronto dei Metodi di Clustering")
    
    if "clustering_results" not in st.session_state or not st.session_state.clustering_results:
        st.warning("Non hai ancora eseguito nessun metodo di clustering. Esplora le sezioni dei singoli metodi prima.")
    elif "clustering_dataset" not in st.session_state:
        st.warning("Non hai ancora generato un dataset. Vai alla sezione 'Dataset Generator'.")
    else:
        st.write("Confronto delle performance di tutti i metodi di clustering eseguiti:")
        
        X = st.session_state.clustering_dataset["X"]
        y_true = st.session_state.clustering_dataset["y_true"]
        
        # Crea tabella comparativa
        comparison_data = {
            "Metodo": [],
            "Silhouette Score": [],
            "Calinski-Harabasz": [],
            "Adjusted Rand Index": [],
            "N. Cluster": []
        }
        
        for method_name, results in st.session_state.clustering_results.items():
            comparison_data["Metodo"].append(method_name)
            comparison_data["Silhouette Score"].append(results["metriche"]["silhouette"])
            comparison_data["Calinski-Harabasz"].append(results["metriche"]["calinski_harabasz"])
            
            if "adjusted_rand" in results["metriche"]:
                comparison_data["Adjusted Rand Index"].append(results["metriche"]["adjusted_rand"])
            else:
                comparison_data["Adjusted Rand Index"].append(np.nan)
            
            # Numero di cluster
            if method_name == "DBSCAN":
                comparison_data["N. Cluster"].append(results["n_clusters"])
            else:
                comparison_data["N. Cluster"].append(len(np.unique(results["labels"])))
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Visualizza tabella
        st.subheader("Tabella Comparativa")
        st.dataframe(df_comparison.style.format({
            "Silhouette Score": "{:.3f}",
            "Calinski-Harabasz": "{:.3f}",
            "Adjusted Rand Index": "{:.3f}"
        }))
        
        # Grafici comparativi
        col1, col2 = st.columns(2)
        
        with col1:
            # Grafico Silhouette Score
            fig_sil = px.bar(df_comparison, x="Metodo", y="Silhouette Score",
                           title="Confronto Silhouette Score",
                           color="Silhouette Score",
                           color_continuous_scale="viridis")
            st.plotly_chart(fig_sil)
        
        with col2:
            # Grafico Calinski-Harabasz
            fig_ch = px.bar(df_comparison, x="Metodo", y="Calinski-Harabasz",
                          title="Confronto Calinski-Harabasz Index",
                          color="Calinski-Harabasz",
                          color_continuous_scale="viridis")
            st.plotly_chart(fig_ch)
        
        # Visualizzazione side-by-side dei risultati
        st.subheader("Visualizzazione Comparativa dei Risultati")
        
        n_methods = len(st.session_state.clustering_results)
        cols = st.columns(min(n_methods, 3))  # Massimo 3 colonne
        
        for i, (method_name, results) in enumerate(st.session_state.clustering_results.items()):
            with cols[i % 3]:
                df_viz = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
                
                if method_name == "DBSCAN":
                    # Gestione speciale per DBSCAN (noise points)
                    df_viz["Cluster"] = results["labels"]
                    df_viz["Cluster_Label"] = df_viz["Cluster"].apply(
                        lambda x: "Noise" if x == -1 else f"C{x}"
                    )
                    color_col = "Cluster_Label"
                else:
                    df_viz["Cluster"] = results["labels"]
                    color_col = "Cluster"
                
                fig = px.scatter(df_viz, x="Feature 1", y="Feature 2", 
                               color=color_col,
                               color_discrete_sequence=px.colors.qualitative.Set1,
                               title=f"{method_name}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Miglior metodo
        if not df_comparison.empty:
            best_silhouette = df_comparison.loc[df_comparison["Silhouette Score"].idxmax()]
            st.success(f"Il metodo con il miglior Silhouette Score è **{best_silhouette['Metodo']}** con un valore di **{best_silhouette['Silhouette Score']:.3f}**.")
        
        # Spiegazione delle metriche
        st.subheader("Spiegazione delle Metriche")
        st.markdown("""
        **Silhouette Score:**
        - Range: [-1, 1]
        - Misura quanto i punti sono simili al proprio cluster vs altri cluster
        - Valori vicini a 1: clustering eccellente
        - Valori vicini a 0: cluster sovrapposti
        - Valori negativi: punti assegnati al cluster sbagliato
        
        **Calinski-Harabasz Index:**
        - Range: [0, ∞)
        - Rapporto tra dispersione inter-cluster e intra-cluster
        - Valori più alti indicano cluster meglio definiti e separati
        
        **Adjusted Rand Index:**
        - Range: [-1, 1] (tipicamente [0, 1])
        - Misura la similarità tra clustering predetto e vero
        - 1: perfetta corrispondenza
        - 0: corrispondenza casuale
        - Disponibile solo quando si conoscono le etichette vere
        """) 
