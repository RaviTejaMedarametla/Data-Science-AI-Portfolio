from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])
    return preprocessor, numeric_cols, categorical_cols


def explain_clusters(df_with_labels: pd.DataFrame, label_col: str) -> dict:
    summary = {}
    for cluster_id, subset in df_with_labels.groupby(label_col):
        cluster_size = len(subset)
        paid_share = float(subset["user_paid"].mean()) if "user_paid" in subset else None
        top_country = subset["onboarding_country"].mode().iloc[0] if "onboarding_country" in subset else None
        top_goal = subset["onboarding_goals_label"].mode().iloc[0] if "onboarding_goals_label" in subset else None
        top_learn = subset["onboarding_learn_label"].mode().iloc[0] if "onboarding_learn_label" in subset else None

        summary[int(cluster_id)] = {
            "size": cluster_size,
            "paid_share": paid_share,
            "top_country": top_country,
            "top_goal": top_goal,
            "top_learning_interest": top_learn,
        }
    return summary




def build_cluster_playbook(cluster_profiles: dict) -> dict:
    """Translate cluster stats into persona + strategy recommendations."""
    playbook = {}

    for cluster_id, profile in cluster_profiles.items():
        paid_share = profile.get("paid_share")
        top_goal = (profile.get("top_goal") or "").lower()
        top_interest = profile.get("top_learning_interest") or ""

        if paid_share is not None and paid_share >= 0.8:
            persona = "Committed Career Accelerator"
            marketing = [
                "Promote advanced learning paths and career-track bundles.",
                "Use outcome-led messaging (promotion, portfolio, interview readiness).",
                "Cross-sell specialization tracks and annual plans.",
            ]
            retention = [
                "Offer milestone nudges tied to goal completion.",
                "Provide personalized next-course recommendations based on learning interest.",
                "Introduce loyalty perks (exclusive webinars, certificate pathways).",
            ]
        elif paid_share is not None and paid_share <= 0.2:
            persona = "Exploring Free Learner"
            marketing = [
                "Run value-focused conversion campaigns with clear paid-benefit comparisons.",
                "Use low-friction trial-to-paid offers and time-limited discounts.",
                "Highlight beginner-friendly guided tracks aligned to their top goal.",
            ]
            retention = [
                "Send re-engagement nudges after inactivity with short actionable lessons.",
                "Use onboarding checklists to create early learning wins.",
                "Recommend entry-level content in their dominant interest area.",
            ]
        else:
            persona = "Hybrid Progress Learner"
            marketing = [
                "Apply segmented messaging by learning goal and country context.",
                "Promote mixed bundles (foundations + intermediate specialization).",
                "Use social proof relevant to their target outcomes.",
            ]
            retention = [
                "Deploy adaptive cadence nudges based on engagement level.",
                "Use cohort/community prompts to sustain consistency.",
                "Provide periodic skill-gap diagnostics with suggested learning plans.",
            ]

        # Light specialization refinement by dominant goal/interest
        if "data scientist" in top_goal:
            marketing.append("Emphasize end-to-end data scientist roadmap messaging.")
            retention.append("Offer capstone/project reminders to reinforce job-readiness progress.")
        if "business analysis" in top_interest.lower():
            marketing.append("Promote BI-focused outcomes and dashboard storytelling modules.")
            retention.append("Trigger dashboard challenge campaigns to keep practical momentum.")

        playbook[int(cluster_id)] = {
            "persona": persona,
            "marketing_strategies": marketing,
            "retention_ideas": retention,
        }

    return playbook


def main() -> None:
    in_path = Path("segmentation_data.csv")
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(in_path)

    # Keep all columns except pure identifier for clustering signal quality
    features_df = df.drop(columns=["user_id"], errors="ignore")

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(features_df)
    X_scaled = preprocessor.fit_transform(features_df)

    # Dense conversion for PCA and agglomerative clustering
    if hasattr(X_scaled, "toarray"):
        X_scaled = X_scaled.toarray()

    # PCA for compact latent space used by clustering algorithms
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Silhouette-based model selection for KMeans
    candidate_k = range(2, 9)
    kmeans_scores = {}
    for k in candidate_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_pca)
        kmeans_scores[k] = float(silhouette_score(X_pca, labels))

    best_k = max(kmeans_scores, key=kmeans_scores.get)
    best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    kmeans_labels = best_kmeans.fit_predict(X_pca)
    kmeans_silhouette = float(silhouette_score(X_pca, kmeans_labels))

    # Hierarchical clustering with same number of clusters for fair comparison
    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hier_labels = hier.fit_predict(X_pca)
    hier_silhouette = float(silhouette_score(X_pca, hier_labels))

    # Attach labels for interpretation
    kmeans_df = df.copy()
    kmeans_df["kmeans_cluster"] = kmeans_labels

    hier_df = df.copy()
    hier_df["hier_cluster"] = hier_labels

    kmeans_explanations = explain_clusters(kmeans_df, "kmeans_cluster")
    hier_explanations = explain_clusters(hier_df, "hier_cluster")

    cluster_playbook = build_cluster_playbook(kmeans_explanations)

    report = {
        "input_shape": list(df.shape),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "pca_n_components": int(pca.n_components_),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "kmeans_silhouette_by_k": kmeans_scores,
        "best_k": int(best_k),
        "kmeans_silhouette": kmeans_silhouette,
        "hierarchical_silhouette": hier_silhouette,
        "kmeans_cluster_profiles": kmeans_explanations,
        "hierarchical_cluster_profiles": hier_explanations,
        "cluster_playbook": cluster_playbook,
        "cluster_interpretation": (
            "Clusters are characterized using paid-share and dominant onboarding attributes "
            "(country, learning goals, learning interests). Use these profiles to tailor "
            "marketing, onboarding nudges, and content pathways by segment."
        ),
    }

    (out_dir / "segmentation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    kmeans_df.to_csv(out_dir / "segmentation_with_kmeans.csv", index=False)
    hier_df.to_csv(out_dir / "segmentation_with_hierarchical.csv", index=False)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
