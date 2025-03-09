import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors

input_dir = "<INPUT_DIR>"
output_dir = "OUTPUT_DIR<>"
os.makedirs(output_dir, exist_ok=True)

methods = ["tsne", "umap", "pca"]
models = ["unet", "resnet", "flattened"]

def compute_cs(real_points, synthetic_points, k=5, threshold=0.05):
    knn = NearestNeighbors(n_neighbors=k).fit(synthetic_points)
    distances, _ = knn.kneighbors(real_points)
    covered = distances[:, 0] < threshold
    return np.sum(covered) / len(real_points)


def compute_pdd(real_points, synthetic_points):
    knn = NearestNeighbors(n_neighbors=1).fit(synthetic_points)
    distances, _ = knn.kneighbors(real_points)
    return distances.flatten()


def compute_pdf(real_points, synthetic_points, r_max=0.5, delta_r=0.005):
    r_values = np.arange(0, r_max, delta_r)
    pdf_values = []
    for r in r_values:
        counts = np.array([np.sum(np.linalg.norm(synthetic_points - p, axis=1) < r) for p in real_points])
        pdf_values.append(counts.mean())
    return r_values, pdf_values


def compute_cha(real_points, synthetic_points):
    hull_real = ConvexHull(real_points)
    hull_synthetic = ConvexHull(synthetic_points)

    from shapely.geometry import Polygon
    poly_real = Polygon(real_points[hull_real.vertices])
    poly_synthetic = Polygon(synthetic_points[hull_synthetic.vertices])

    if poly_real.intersects(poly_synthetic):
        poly_overlap = poly_real.intersection(poly_synthetic)
        a_overlap = poly_overlap.area
    else:
        a_overlap = 0

    return hull_real.volume, hull_synthetic.volume, a_overlap
    hull_real = ConvexHull(real_points)
    hull_synthetic = ConvexHull(synthetic_points)
    combined_points = np.vstack([real_points, synthetic_points])
    hull_combined = ConvexHull(combined_points)
    return hull_real.volume, hull_synthetic.volume, hull_combined.volume


def compute_ood(real_points, synthetic_points):
    mean_real = np.mean(real_points, axis=0)
    cov_real = EmpiricalCovariance().fit(real_points)
    mahal_distances = np.sqrt(np.sum((synthetic_points - mean_real) @ np.linalg.inv(cov_real.covariance_) * (synthetic_points - mean_real), axis=1))
    d_95 = np.percentile(mahal_distances, 95)
    ood_rate = round(np.sum(mahal_distances > d_95) / len(synthetic_points), 5)
    return mahal_distances, d_95, ood_rate

cs_results = []
cha_results = []
ood_stats = []

for method in methods:
    for model in models:
        file_path = os.path.join(input_dir, f"{method}_{model}_features.csv")
        df = pd.read_csv(file_path)

        df['Dimension 1'] = (df['Dimension 1'] - df['Dimension 1'].min()) / (
                    df['Dimension 1'].max() - df['Dimension 1'].min())
        df['Dimension 2'] = (df['Dimension 2'] - df['Dimension 2'].min()) / (
                    df['Dimension 2'].max() - df['Dimension 2'].min())

        real_points = df[df['Label'] == 0][['Dimension 1', 'Dimension 2']].values
        synthetic_points = df[df['Label'] == 1][['Dimension 1', 'Dimension 2']].values

        # CS
        cs_score = compute_cs(real_points, synthetic_points)
        cs_results.append([method, model, cs_score])

        # PDD
        pdd_values = compute_pdd(real_points, synthetic_points)
        pd.DataFrame(pdd_values, columns=["NND"]).to_csv(
            os.path.join(output_dir, f"{method}_{model}_pairwise_distance.csv"), index=False)
        plt.figure()
        sns.histplot(pdd_values, kde=True)
        plt.title(f"PDD Histogram ({method}-{model})")
        plt.savefig(os.path.join(output_dir, f"{method}_{model}_pdd_histogram.png"))
        plt.close()

        # PDF
        r_vals, g_vals = compute_pdf(real_points, synthetic_points)
        pd.DataFrame({"r": r_vals, "G(r)": g_vals}).to_csv(
            os.path.join(output_dir, f"{method}_{model}_pair_distribution_function.csv"), index=False)
        plt.figure()
        plt.plot(r_vals, g_vals)
        plt.fill_between(r_vals, np.array(g_vals) - np.std(g_vals), np.array(g_vals) + np.std(g_vals), alpha=0.3)
        plt.title(f"PDF Curve ({method}-{model})")
        plt.savefig(os.path.join(output_dir, f"{method}_{model}_pdf_curve.png"))
        plt.close()

        # CHA
        a_r, a_s, a_overlap = compute_cha(real_points, synthetic_points)
        cha_results.append([method, model, a_r, a_s, a_overlap, a_overlap / a_r])

        # OOD
        mahal_distances, d_95, ood_rate = compute_ood(real_points, synthetic_points)
        pd.DataFrame(mahal_distances, columns=["Mahalanobis Distance"]).to_csv(
            os.path.join(output_dir, f"{method}_{model}_ood_distance.csv"), index=False)
        ood_stats.append([method, model, d_95, ood_rate])
        plt.figure()
        sns.histplot(mahal_distances, kde=True)
        plt.title(f"OOD Histogram ({method}-{model})")
        plt.savefig(os.path.join(output_dir, f"{method}_{model}_ood_histogram.png"))
        plt.close()

pd.DataFrame(cs_results, columns=["Method", "Model", "Coverage Score"]).to_csv(
    os.path.join(output_dir, "coverage_score.csv"), index=False)

pd.DataFrame(cha_results, columns=["Method", "Model", "A_R", "A_S", "A_Overlap", "CHA Ratio"]).to_csv(
    os.path.join(output_dir, "convex_hull_area.csv"), index=False)

pd.DataFrame(ood_stats, columns=["Method", "Model", "D_95", "OOD Expansion Rate"]).to_csv(
    os.path.join(output_dir, "ood_stat.csv"), index=False)
