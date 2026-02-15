"""
Phase 5 Task 5.2: Additional Visualizations
Creates visualizations for:
- Predicted vs Actual Rating Distribution
- Embedding Visualization (t-SNE)
- User Segmentation Performance
- Category Cross-Recommendations
- Long-Tail Analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.manifold import TSNE
from utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def plot_predicted_vs_actual_ratings(save_path: str = 'results/visualizations/predicted_vs_actual.png'):
    """
    Visualization 4: Predicted vs Actual Rating Distribution
    """
    logger.info("Creating predicted vs actual rating distribution...")

    # Load test data
    test_df = pd.read_csv('dataset/test_set.csv')

    # Load MF model
    from utils.model_loader import load_mf_model
    mf_model = load_mf_model()

    # Load mappings
    with open('dataset/user_to_idx.pkl', 'rb') as f:
        user_to_idx = pickle.load(f)
    with open('dataset/item_to_idx.pkl', 'rb') as f:
        item_to_idx = pickle.load(f)

    # Get predictions for test set
    predictions = []
    actuals = []

    for _, row in test_df.iterrows():
        user_id = row['User.ID']
        clothing_id = row['Clothing.ID']

        if user_id in user_to_idx and clothing_id in item_to_idx:
            user_idx = user_to_idx[user_id]
            item_idx = item_to_idx[clothing_id]

            pred = mf_model.predict(user_idx, item_idx)
            predictions.append(pred)
            actuals.append(row['Rating'])

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Histograms
    axes[0].hist(actuals, bins=5, alpha=0.6, label='Actual', color='blue', range=(1, 5))
    axes[0].hist(predictions, bins=20, alpha=0.6, label='Predicted', color='red', range=(1, 5))
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Predicted vs Actual Rating Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Scatter plot
    # Sample for readability
    sample_size = min(1000, len(predictions))
    idx = np.random.choice(len(predictions), sample_size, replace=False)

    axes[1].scatter(actuals[idx], predictions[idx], alpha=0.3, s=20)
    axes[1].plot([1, 5], [1, 5], 'r--', label='Perfect Prediction')
    axes[1].set_xlabel('Actual Rating')
    axes[1].set_ylabel('Predicted Rating')
    axes[1].set_title('Predicted vs Actual Ratings (Sample)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0.5, 5.5)
    axes[1].set_ylim(0.5, 5.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved predicted vs actual plot: {save_path}")

    # Print statistics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")


def plot_embedding_tsne(save_path: str = 'results/visualizations/embedding_tsne.png'):
    """
    Visualization 5: Embedding Visualization (t-SNE)
    """
    logger.info("Creating t-SNE embedding visualization...")

    # Load MF model
    from utils.model_loader import load_mf_model
    mf_model = load_mf_model()

    # Get item embeddings
    item_embeddings = mf_model.get_all_item_embeddings()

    # Load item features for labels
    item_features = pd.read_csv('dataset/item_features.csv')

    # Align embeddings with features
    with open('dataset/item_to_idx.pkl', 'rb') as f:
        item_to_idx = pickle.load(f)

    # Create mapping
    clothing_ids = []
    departments = []
    embeddings = []

    for _, row in item_features.iterrows():
        clothing_id = row['Clothing.ID']
        if clothing_id in item_to_idx:
            idx = item_to_idx[clothing_id]
            clothing_ids.append(clothing_id)
            departments.append(row['department'])
            embeddings.append(item_embeddings[idx])

    embeddings = np.array(embeddings)

    # Apply t-SNE
    logger.info(f"Running t-SNE on {len(embeddings)} item embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get unique departments and colors
    unique_depts = sorted(set(departments))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_depts)))
    dept_to_color = {dept: colors[i] for i, dept in enumerate(unique_depts)}

    # Plot each department
    for dept in unique_depts:
        mask = np.array(departments) == dept
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=dept,
            alpha=0.6,
            s=50,
            c=[dept_to_color[dept]]
        )

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization of Item Embeddings by Department')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved t-SNE plot: {save_path}")


def plot_user_segmentation_performance(save_path: str = 'results/visualizations/user_segmentation_performance.png'):
    """
    Visualization 6: User Segmentation Performance
    """
    logger.info("Creating user segmentation performance plot...")

    # Load segment analysis results
    segment_df = pd.read_csv('results/segment_analysis/user_segment_analysis.csv')

    # Create grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    segments = ['New', 'Casual', 'Active']
    metrics = ['precision@10', 'ndcg@10', 'hit_rate@10']
    metric_names = ['Precision@10', 'NDCG@10', 'Hit Rate@10']

    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]

        # Prepare data
        models = segment_df['model'].unique()
        x = np.arange(len(segments))
        width = 0.25

        for j, model in enumerate(models):
            model_data = segment_df[segment_df['model'] == model]
            values = [
                model_data[model_data['segment'] == seg][metric].values[0]
                if len(model_data[model_data['segment'] == seg]) > 0 else 0
                for seg in segments
            ]
            ax.bar(x + j * width, values, width, label=model)

        ax.set_xlabel('User Segment')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by User Segment')
        ax.set_xticks(x + width)
        ax.set_xticklabels(segments)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved user segmentation performance plot: {save_path}")


def plot_category_cross_recommendations(save_path: str = 'results/visualizations/category_cross_recommendations.png'):
    """
    Visualization 7: Category Cross-Recommendations Heatmap
    Shows: rows=user's favorite dept, columns=recommended dept
    """
    logger.info("Creating category cross-recommendations heatmap...")

    # Load necessary data
    test_df = pd.read_csv('dataset/test_set.csv')
    user_features = pd.read_csv('dataset/user_features.csv')
    item_features = pd.read_csv('dataset/item_features.csv')

    # Load MF model
    from utils.model_loader import load_mf_model
    mf_model = load_mf_model()

    # Load mappings
    with open('dataset/user_to_idx.pkl', 'rb') as f:
        user_to_idx = pickle.load(f)
    with open('dataset/item_to_idx.pkl', 'rb') as f:
        item_to_idx = pickle.load(f)
    idx_to_item = {v: k for k, v in item_to_idx.items()}

    # Create clothing_id to department mapping
    item_to_dept = {}
    for _, row in item_features.iterrows():
        item_to_dept[row['Clothing.ID']] = row['department']

    # Create user to favorite department mapping
    user_to_fav_dept = {}
    for _, row in user_features.iterrows():
        user_to_fav_dept[row['User.ID']] = row['favorite_department']

    # Count cross-recommendations
    departments = sorted(item_features['department'].unique())
    cross_matrix = np.zeros((len(departments), len(departments)))
    dept_to_idx = {dept: i for i, dept in enumerate(departments)}

    # Sample test users
    test_users = test_df['User.ID'].unique()[:100]  # Sample for efficiency

    for user_id in test_users:
        if user_id not in user_to_idx or user_id not in user_to_fav_dept:
            continue

        user_idx = user_to_idx[user_id]
        fav_dept = user_to_fav_dept[user_id]

        if fav_dept not in dept_to_idx:
            continue

        # Get top-10 recommendations
        n_items = len(item_to_idx)
        item_indices = np.arange(n_items)
        predictions = mf_model.predict_batch(user_idx, item_indices)

        # Mask training items
        train_matrix = load_npz('dataset/train_matrix.npz')
        train_items = np.where(train_matrix[user_idx].toarray().flatten() > 0)[0]
        predictions[train_items] = -np.inf

        top_k_items = np.argsort(predictions)[::-1][:10]

        # Count department of recommendations
        fav_idx = dept_to_idx[fav_dept]
        for item_idx in top_k_items:
            clothing_id = idx_to_item[item_idx]
            if clothing_id in item_to_dept:
                rec_dept = item_to_dept[clothing_id]
                if rec_dept in dept_to_idx:
                    rec_idx = dept_to_idx[rec_dept]
                    cross_matrix[fav_idx, rec_idx] += 1

    # Normalize by row
    row_sums = cross_matrix.sum(axis=1, keepdims=True)
    cross_matrix_norm = np.divide(cross_matrix, row_sums, where=row_sums != 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cross_matrix_norm,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=departments,
        yticklabels=departments,
        cbar_kws={'label': 'Proportion'},
        ax=ax
    )

    ax.set_xlabel('Recommended Department')
    ax.set_ylabel("User's Favorite Department")
    ax.set_title('Category Cross-Recommendation Pattern')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved category cross-recommendations heatmap: {save_path}")


def plot_long_tail_analysis(save_path: str = 'results/visualizations/long_tail_analysis.png'):
    """
    Visualization 8: Long-Tail Analysis
    Scatter plot: X=item popularity, Y=times recommended, color=model
    """
    logger.info("Creating long-tail analysis plot...")

    # Load item features
    item_features = pd.read_csv('dataset/item_features.csv')

    # Load models
    from utils.model_loader import load_all_models

    pop_model, mf_model, ncf_model = load_all_models()

    models = [
        (pop_model, 'Popularity', 'blue'),
        (mf_model, 'MF', 'green'),
        (ncf_model, 'NCF', 'red')
    ]

    # Load mappings and test data
    with open('dataset/user_to_idx.pkl', 'rb') as f:
        user_to_idx = pickle.load(f)
    with open('dataset/item_to_idx.pkl', 'rb') as f:
        item_to_idx = pickle.load(f)
    idx_to_item = {v: k for k, v in item_to_idx.items()}

    test_df = pd.read_csv('dataset/test_set.csv')
    test_users = test_df['User.ID'].unique()

    # Count recommendations per model
    fig, ax = plt.subplots(figsize=(12, 8))

    for model, model_name, color in models:
        item_rec_count = {clothing_id: 0 for clothing_id in item_to_idx.keys()}

        # Generate recommendations for all test users
        for user_id in test_users:
            if user_id not in user_to_idx:
                continue

            user_idx = user_to_idx[user_id]
            n_items = len(item_to_idx)
            item_indices = np.arange(n_items)

            # Get predictions
            if hasattr(model, 'predict_batch'):
                predictions = model.predict_batch(user_idx, item_indices)
            elif hasattr(model, 'get_scores'):
                predictions = model.get_scores(user_idx)
            else:
                continue

            # Mask training items
            train_matrix = load_npz('dataset/train_matrix.npz')
            train_items = np.where(train_matrix[user_idx].toarray().flatten() > 0)[0]
            predictions[train_items] = -np.inf

            # Top-10
            top_k_items = np.argsort(predictions)[::-1][:10]

            for item_idx in top_k_items:
                clothing_id = idx_to_item[item_idx]
                item_rec_count[clothing_id] += 1

        # Prepare data for plotting
        popularities = []
        rec_counts = []

        for _, row in item_features.iterrows():
            clothing_id = row['Clothing.ID']
            if clothing_id in item_rec_count:
                popularities.append(row['rating_count'])
                rec_counts.append(item_rec_count[clothing_id])

        # Plot
        ax.scatter(
            popularities,
            rec_counts,
            alpha=0.5,
            s=30,
            label=model_name,
            c=color
        )

    ax.set_xlabel('Item Popularity (Review Count)')
    ax.set_ylabel('Times Recommended')
    ax.set_title('Long-Tail Analysis: Item Popularity vs Recommendation Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved long-tail analysis plot: {save_path}")


def main():
    """Create all additional visualizations."""
    logger.info("="*60)
    logger.info("PHASE 5 TASK 5.2: ADDITIONAL VISUALIZATIONS")
    logger.info("="*60)

    output_dir = Path('results/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualization 4: Predicted vs Actual
    logger.info("\n[1/5] Predicted vs Actual Rating Distribution...")
    plot_predicted_vs_actual_ratings()

    # Visualization 5: t-SNE Embeddings
    logger.info("\n[2/5] t-SNE Embedding Visualization...")
    plot_embedding_tsne()

    # Visualization 6: User Segmentation Performance
    logger.info("\n[3/5] User Segmentation Performance...")
    plot_user_segmentation_performance()

    # Visualization 7: Category Cross-Recommendations
    logger.info("\n[4/5] Category Cross-Recommendations...")
    plot_category_cross_recommendations()

    # Visualization 8: Long-Tail Analysis
    logger.info("\n[5/5] Long-Tail Analysis...")
    plot_long_tail_analysis()

    logger.info("\n" + "="*60)
    logger.info("ALL VISUALIZATIONS COMPLETE!")
    logger.info(f"Saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
