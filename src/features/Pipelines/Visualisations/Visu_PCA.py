from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd


def afficher_pca(pca_ref, data_x, data_flat, data_pca):
        """Affiche les résultats de l'analyse PCA.
        
        Args:
            data_x: Images originales
            data_flat: Images aplaties avant PCA
            data_pca: Images transformées par PCA
        """
        print(f"\n🔬 Analyse en Composantes Principales (PCA)")
        print(f"📊 Nombre de composantes: {pca_ref.n_components}")
        
        # Variance expliquée par chaque composante
        explained_variance_ratio = pca_ref.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Créer un DataFrame pour les composantes PCA
        n_show = min(20, pca_ref.n_components)  # Augmenté à 20 pour plus de détails
        
        pca_df = pd.DataFrame({
            'Composante': [f'PC{i+1}' for i in range(n_show)],
            'Variance_Expliquée': explained_variance_ratio[:n_show],
            'Variance_Expliquée_%': (explained_variance_ratio[:n_show] * 100),
            'Variance_Cumulée': cumulative_variance[:n_show],
            'Variance_Cumulée_%': (cumulative_variance[:n_show] * 100)
        })
        
        # Formatage pour un affichage plus propre
        pca_df = pca_df.round({
            'Variance_Expliquée': 4,
            'Variance_Expliquée_%': 2,
            'Variance_Cumulée': 4,
            'Variance_Cumulée_%': 2
        })
        
        print(f"\n📈 Variance expliquée par composante:")
        
        # Utiliser display() pour un affichage interactif du DataFrame dans les notebooks
        try:
            from IPython.display import display
            display(pca_df)
        except ImportError:
            # Fallback si IPython n'est pas disponible (ex: script Python classique)
            print(pca_df.to_string(index=False, 
                                   col_space={'Composante': 12, 
                                            'Variance_Expliquée': 18,
                                            'Variance_Expliquée_%': 20,
                                            'Variance_Cumulée': 16,
                                            'Variance_Cumulée_%': 18}))
        
        if pca_ref.n_components > n_show:
            print(f"\n... ({pca_ref.n_components - n_show} composantes supplémentaires)")
        
        print(f"\n📈 Variance totale expliquée: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.2f}%)")
                
        # Graphique de la variance expliquée
        plt.figure(figsize=(12, 4))
        
        # Subplot 1: Variance par composante
        plt.subplot(1, 2, 1)
        plt.bar(range(1, min(21, pca_ref.n_components + 1)), 
                explained_variance_ratio[:min(20, pca_ref.n_components)], 
                alpha=0.7, color='steelblue')
        plt.xlabel('Composante Principale')
        plt.ylabel('Variance Expliquée')
        plt.title('Variance Expliquée par Composante')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Variance cumulée
        plt.subplot(1, 2, 2)
        plt.plot(range(1, pca_ref.n_components + 1), 
                 cumulative_variance, 
                 'o-', color='red', linewidth=2, markersize=4)
        plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% variance')
        plt.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% variance')
        plt.xlabel('Nombre de Composantes')
        plt.ylabel('Variance Cumulée')
        plt.title('Variance Cumulée Expliquée')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        """
        # Visualisation des composantes principales (pas des données transformées!)
        original_shape = data_x.shape
        if len(original_shape) >= 3:  # Images 2D ou plus
            # Déterminer les dimensions d'image à partir des données originales
            if len(original_shape) == 3:  # (n_samples, height, width)
                img_height, img_width = original_shape[1], original_shape[2]
            elif len(original_shape) == 4:  # (n_samples, height, width, channels)
                img_height, img_width = original_shape[1], original_shape[2]
            else:
                # Essayer de deviner les dimensions à partir du nombre total de pixels
                total_pixels = data_flat.shape[1]
                img_height = img_width = int(np.sqrt(total_pixels))
            
            print(f"\n🖼️ Visualisation des composantes principales:")
            print(f"📐 Forme originale des images: {img_height}x{img_width}")
            n_components_to_show = min(6, pca_ref.n_components)
            
            plt.figure(figsize=(15, 3))
            for i in range(n_components_to_show):
                plt.subplot(1, n_components_to_show, i + 1)
                
                # CORRECTION: Utiliser les composantes du PCA, pas les données transformées
                try:
                    # Les composantes principales ont la même dimension que les données aplaties
                    component_img = pca_ref.pca.components_[i].reshape(img_height, img_width)
                    plt.imshow(component_img, cmap='RdBu_r', aspect='equal')
                    plt.title(f'PC{i+1}\n({explained_variance_ratio[i]:.1%})')
                    plt.axis('off')
                except ValueError as e:
                    plt.text(0.5, 0.5, f'Erreur\nreshape\n{str(e)[:30]}...', 
                            ha='center', va='center', transform=plt.gca().transAxes,
                            fontsize=8)
                    plt.title(f'PC{i+1}')
                    plt.axis('off')
            
            plt.suptitle('Composantes Principales (comme images)', fontsize=14)
            plt.tight_layout()
            plt.show()   """
             


def create_interactive_pca_plot(pca_ref,data_x, data_pca):
    """Crée un graphique interactif Plotly avec images au survol."""
    explained_variance_ratio = pca_ref.explained_variance_ratio_  
    indices = np.arange(len(data_x))
        
    # Graphique 3D si disponible
    if pca_ref.n_components >= 3:
        print(f"\n🎲 Visualisation 3D de l'espace PCA:")

        fig_3d = go.Figure(data=[go.Scatter3d(
        x=data_pca[:, 0],
        y=data_pca[:, 1],
        z=data_pca[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=np.arange(len(data_pca)),
            colorscale='Viridis',
            showscale=True
        ),
        hovertemplate='<b>Image %{text}</b><br>' +
                     'PC1: %{x:.3f}<br>' +
                     'PC2: %{y:.3f}<br>' +
                     'PC3: %{z:.3f}<br>' +
                     '<extra></extra>',
        text=indices,
        name='Images PCA 3D'
        )])        

        fig_3d.update_layout(
        title=f'🎲 Espace PCA 3D Interactif',
        scene=dict(
            xaxis_title=f'PC1 ({explained_variance_ratio[0]:.1%})',
            yaxis_title=f'PC2 ({explained_variance_ratio[1]:.1%})',
            zaxis_title=f'PC3 ({explained_variance_ratio[2]:.1%})'
            ),
            width=900,
            height=700
        )
            
        fig_3d.show()
        
    print("\n✅ Visualisations interactives créées !")
    print("\n💡 Utilisez le slider ci-dessus pour explorer les images et leur position PCA\n")