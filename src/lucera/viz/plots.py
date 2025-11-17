import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import habitat_pl.viz.corine as corine

def make_embeddings_plots(embedding_pca, embedding_tsne, embedding_umap, labels):

    # --- Color mapping (same as before) ---
    code_to_color = {}
    for k, v in corine.legend_dict.items():
        code = k.split()[0]           # e.g. "111"
        code_to_color[code] = f"#{v}" # add '#' for matplotlib

    colors = [code_to_color.get(str(label), "#999999") for label in labels]  # grey fallback

    # --- Prepare subplots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ["PCA projection", "t-SNE projection", "UMAP projection"]
    embeddings = [embedding_pca, embedding_tsne, embedding_umap]

    for ax, emb, title in zip(axes, embeddings, titles):
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            s=10,
            c=colors,
            alpha=0.7
        )
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_box_aspect(1)       # square aspect ratio

    # --- Shared legend (same colors as scatter) ---
    unique_labels = np.unique(labels)
    patches = [
        mpatches.Patch(color=code_to_color.get(str(lbl), "#999999"), label=str(lbl))
        for lbl in unique_labels
    ]

    fig.legend(
        handles=patches,
        title="CORINE class",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.
    )

    plt.tight_layout()
    plt.show()

def add_umap_class(X, inliers, outliers, L, ax):

    color = [v for k,v in corine.legend_dict.items() if k.split()[0]==str(L)][0]
    
    ax.scatter(
        X[inliers, 0], X[inliers, 1],
        s=10, c=f"#{color}", alpha=0.7
    )

    ax.scatter(
        X[outliers, 0], X[outliers, 1],
        s=10, c="grey", alpha=0.02
    )


def make_umap_plot(embedding_umap, labels, alpha=0.7, ax=None, legend=False):

    # Plot the result
    if not ax:
        plt.figure(figsize=(8, 8))
        fig, ax = plt.subplots()
        #legend = True

    code_to_color = {}
    for k, v in corine.legend_dict.items():
        code = k.split()[0]           # e.g. "111" from "111 Continuous urban fabric"
        code_to_color[code] = f"#{v}" # add '#' for matplotlib

    colors = [code_to_color.get(str(label), "#999999") for label in labels]  # grey fallback
    colors = np.array(colors)

    ax.scatter(
        embedding_umap[:, 0],
        embedding_umap[:, 1],
        s=10,
        c=colors,
        alpha = alpha
    )

    if False: #emlabel:
        eml = emlabel
        labs = np.array(labels).reshape((-1,))
        print(embedding_umap[labs==eml].shape)
        ax.scatter(
            embedding_umap[labs==eml][:, 0],
            embedding_umap[labs==eml][:, 1],
            s=10,
            c=colors[labs==eml],
            alpha = 0.7
        )

    unique_labels = np.unique(labels)
    patches = [
        mpatches.Patch(color=code_to_color.get(str(lbl), "#999999"), label=str(lbl))
        for lbl in unique_labels
    ]

    if legend:
        plt.legend(
            handles=patches,
            title="CORINE class",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.
        )

    plt.title("UMAP projection of 64-D embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    #ax.set_xlim(-2.5,12.5)
    #ax.set_ylim(-1,9)
    ax.set_box_aspect(1)
    plt.tight_layout()

    return ax
    #plt.show()