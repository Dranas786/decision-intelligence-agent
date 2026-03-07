from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from app.analytics.common import build_insight, build_result

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None


DEFAULT_SEVERITY_BANDS = {
    "minor": 0.02,
    "moderate": 0.05,
    "severe": 0.08,
}


@dataclass
class PointCloudData:
    points: np.ndarray
    normals: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    def copy(self) -> "PointCloudData":
        return PointCloudData(
            points=self.points.copy(),
            normals=None if self.normals is None else self.normals.copy(),
            metadata=dict(self.metadata),
            source_path=self.source_path,
        )


@dataclass
class PipeFit:
    axis: np.ndarray
    center: np.ndarray
    radius: float
    residuals: np.ndarray
    axial_positions: np.ndarray
    angles: np.ndarray
    radial_vectors: np.ndarray
    radial_distances: np.ndarray
    fit_rmse: float
    fit_mae: float


@dataclass
class DeviationMap:
    deviations: np.ndarray
    axial_positions: np.ndarray
    angles: np.ndarray
    coordinates: np.ndarray
    nominal_radius: float
    fit_rmse: float
    units: str



def load_point_cloud(dataset_path: str) -> PointCloudData:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = path.suffix.lower()
    if suffix in {".xyz", ".txt"}:
        raw = np.loadtxt(path)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] < 3:
            raise ValueError("XYZ point clouds must contain at least three columns.")
        points = raw[:, :3].astype(float)
        if len(points) == 0:
            raise ValueError(f"Point cloud '{dataset_path}' contains no points.")
        return PointCloudData(points=points, metadata={"format": suffix}, source_path=str(path))

    if suffix in {".ply", ".pcd"}:
        if o3d is None:
            raise ImportError("Open3D is required to read PLY or PCD point clouds.")
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=float)
        normals = np.asarray(cloud.normals, dtype=float) if cloud.has_normals() else None
        if len(points) == 0:
            raise ValueError(f"Point cloud '{dataset_path}' contains no points.")
        return PointCloudData(points=points, normals=normals, metadata={"format": suffix}, source_path=str(path))

    raise ValueError("Unsupported point-cloud file type. Use PLY, PCD, or XYZ.")



def profile_point_cloud(cloud: PointCloudData, units: str = "unitless") -> dict[str, Any]:
    points = cloud.points
    point_count = int(len(points))
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    extent = max_bounds - min_bounds
    bbox_volume = float(np.prod(np.maximum(extent, 1e-9)))
    density_proxy = float(point_count / bbox_volume) if bbox_volume else float(point_count)
    has_normals = cloud.normals is not None and len(cloud.normals) == point_count

    data = {
        "point_count": point_count,
        "bounds": {"min": min_bounds.tolist(), "max": max_bounds.tolist()},
        "extent": extent.tolist(),
        "density_proxy": density_proxy,
        "has_normals": has_normals,
        "units": units,
        "format": cloud.metadata.get("format"),
    }
    return build_result(
        insights=[f"Point cloud has {point_count} points with extent {extent.round(4).tolist()} {units}."],
        insight_objects=[
            build_insight(
                tool="profile_point_cloud",
                title="Point cloud profile",
                message="Basic point-cloud profiling completed.",
                evidence=data,
            )
        ],
        data=data,
        charts=[{"type": "bounds", "title": "Point-cloud bounds"}],
    )



def clean_point_cloud(
    cloud: PointCloudData,
    voxel_size: float = 0.0,
    max_outlier_std: float = 2.0,
    neighbors: int = 20,
    estimate_normals: bool = True,
) -> dict[str, Any]:
    points = cloud.points.copy()
    diagnostics: list[str] = []

    if voxel_size and voxel_size > 0:
        points = _voxel_downsample(points, voxel_size)

    if len(points) >= max(neighbors + 1, 4):
        mask = _inlier_mask(points, neighbors=neighbors, std_ratio=max_outlier_std)
        points = points[mask]
    else:
        diagnostics.append("Point cloud too small for statistical outlier removal; kept all points.")

    normals = None
    if estimate_normals and len(points) >= 6:
        normals = _estimate_normals(points, neighbor_count=min(neighbors, max(6, len(points) - 1)))
    elif cloud.normals is not None and len(cloud.normals) == len(points):
        normals = cloud.normals.copy()
    else:
        diagnostics.append("Normals were not estimated for the cleaned cloud.")

    cleaned = PointCloudData(
        points=points,
        normals=normals,
        metadata={**cloud.metadata, "voxel_size": voxel_size, "max_outlier_std": max_outlier_std},
        source_path=cloud.source_path,
    )

    data = {
        "original_points": int(len(cloud.points)),
        "clean_points": int(len(points)),
        "voxel_size": voxel_size,
        "max_outlier_std": max_outlier_std,
        "has_normals": normals is not None,
    }
    return build_result(
        insights=[f"Cleaned point cloud from {len(cloud.points)} to {len(points)} points."],
        insight_objects=[
            build_insight(
                tool="clean_point_cloud",
                title="Point cloud cleaning",
                message="Downsampling and outlier removal completed.",
                evidence=data,
            )
        ],
        diagnostics=diagnostics,
        data=data,
        artifacts={"point_cloud": cleaned},
    )



def fit_pipe_cylinder(
    cloud: PointCloudData,
    axis_hint: list[float] | tuple[float, float, float] | None = None,
    expected_radius: float | None = None,
    units: str = "unitless",
) -> dict[str, Any]:
    if len(cloud.points) < 12:
        return build_result(
            insights=["Cylinder fitting skipped because the point cloud is too sparse."],
            diagnostics=["Cylinder fitting requires at least 12 points."],
        )

    axis = _resolve_axis(cloud.points, axis_hint)
    if axis is None:
        return build_result(
            insights=["Cylinder fitting failed because a stable axis could not be determined."],
            diagnostics=["Axis hint is invalid or the point distribution is degenerate."],
        )

    b1, b2 = _orthonormal_basis(axis)
    centroid = cloud.points.mean(axis=0)
    centered = cloud.points - centroid
    plane_coords = np.column_stack((centered @ b1, centered @ b2))
    circle_center_2d, fitted_radius = _fit_circle_2d(plane_coords)
    center = centroid + (b1 * circle_center_2d[0]) + (b2 * circle_center_2d[1])

    if expected_radius and expected_radius > 0:
        fitted_radius = float((fitted_radius + expected_radius) / 2.0)

    axial_positions, radial_vectors, radial_distances, angles = _cylindrical_coordinates(cloud.points, center, axis, b1, b2)
    residuals = radial_distances - fitted_radius
    fit_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    fit_mae = float(np.mean(np.abs(residuals)))

    fit = PipeFit(
        axis=axis,
        center=center,
        radius=float(fitted_radius),
        residuals=residuals,
        axial_positions=axial_positions,
        angles=angles,
        radial_vectors=radial_vectors,
        radial_distances=radial_distances,
        fit_rmse=fit_rmse,
        fit_mae=fit_mae,
    )

    data = {
        "axis": axis.tolist(),
        "center": center.tolist(),
        "radius": float(fitted_radius),
        "fit_rmse": fit_rmse,
        "fit_mae": fit_mae,
        "axial_span": [float(axial_positions.min()), float(axial_positions.max())],
        "units": units,
    }

    diagnostics: list[str] = []
    if expected_radius and expected_radius > 0:
        diagnostics.append(f"Expected radius prior was supplied: {expected_radius:.4f} {units}.")
    if fit_rmse > max(fitted_radius * 0.25, 1e-6):
        diagnostics.append("Fit residual is high relative to radius; downstream dent metrics may be unreliable.")

    return build_result(
        insights=[f"Fitted nominal cylinder radius {fitted_radius:.4f} {units} with RMSE {fit_rmse:.4f}."],
        insight_objects=[
            build_insight(
                tool="fit_pipe_cylinder",
                title="Cylinder fit",
                message="Nominal pipe cylinder fit completed.",
                evidence=data,
            )
        ],
        diagnostics=diagnostics,
        data=data,
        artifacts={"pipe_fit": fit},
        charts=[{"type": "axial_profile", "title": "Cylinder fit residual profile"}],
    )



def compute_pipe_deviation_map(
    cloud: PointCloudData,
    pipe_fit: PipeFit,
    units: str = "unitless",
) -> dict[str, Any]:
    deviations = pipe_fit.residuals.copy()
    deviation_map = DeviationMap(
        deviations=deviations,
        axial_positions=pipe_fit.axial_positions.copy(),
        angles=pipe_fit.angles.copy(),
        coordinates=cloud.points.copy(),
        nominal_radius=float(pipe_fit.radius),
        fit_rmse=float(pipe_fit.fit_rmse),
        units=units,
    )

    inward = deviations[deviations < 0]
    outward = deviations[deviations > 0]
    data = {
        "nominal_radius": float(pipe_fit.radius),
        "min_deviation": float(deviations.min()),
        "max_deviation": float(deviations.max()),
        "mean_inward_deviation": float(inward.mean()) if len(inward) else 0.0,
        "mean_outward_deviation": float(outward.mean()) if len(outward) else 0.0,
        "fit_rmse": float(pipe_fit.fit_rmse),
        "units": units,
    }

    return build_result(
        insights=[f"Computed deviation map with minimum radial deviation {deviations.min():.4f} {units}."],
        insight_objects=[
            build_insight(
                tool="compute_pipe_deviation_map",
                title="Deviation map",
                message="Signed radial deviations from the nominal cylinder were computed.",
                evidence=data,
            )
        ],
        data=data,
        artifacts={"deviation_map": deviation_map},
        charts=[
            {"type": "heatmap", "title": "Deviation heatmap"},
            {"type": "axial_profile", "title": "Axial deviation profile"},
        ],
    )



def detect_pipe_dents(
    cloud: PointCloudData,
    deviation_map: DeviationMap,
    deviation_threshold: float,
    min_cluster_points: int = 20,
    severity_bands: dict[str, float] | None = None,
) -> dict[str, Any]:
    severity_bands = severity_bands or DEFAULT_SEVERITY_BANDS
    candidate_mask = deviation_map.deviations <= -abs(deviation_threshold)
    candidate_points = deviation_map.coordinates[candidate_mask]

    if len(candidate_points) < min_cluster_points:
        return build_result(
            insights=["No dents detected above the configured inward-deviation threshold."],
            data={"dents": []},
            diagnostics=["Insufficient candidate points remained after thresholding."],
            charts=[{"type": "heatmap", "title": "Dent candidates"}],
        )

    eps = _cluster_radius(candidate_points)
    labels = DBSCAN(eps=eps, min_samples=max(3, min_cluster_points // 4)).fit_predict(candidate_points)
    candidate_indices = np.where(candidate_mask)[0]
    dent_records: list[dict[str, Any]] = []
    insight_objects: list[dict[str, Any]] = []
    insights: list[str] = []

    for label in sorted(set(labels)):
        if label == -1:
            continue
        label_mask = labels == label
        if int(label_mask.sum()) < min_cluster_points:
            continue

        global_indices = candidate_indices[label_mask]
        coords = deviation_map.coordinates[global_indices]
        cluster_deviations = deviation_map.deviations[global_indices]
        axial = deviation_map.axial_positions[global_indices]
        angles = deviation_map.angles[global_indices]
        depth = float(abs(cluster_deviations.min()))
        dent = {
            "dent_id": f"dent_{len(dent_records) + 1}",
            "depth": depth,
            "axial_start": float(axial.min()),
            "axial_end": float(axial.max()),
            "axial_length": float(axial.max() - axial.min()),
            "circumferential_width": float(_angular_span(angles) * deviation_map.nominal_radius),
            "point_count": int(len(global_indices)),
            "severity": _severity_for_depth(depth, deviation_map.nominal_radius, severity_bands),
            "centroid": coords.mean(axis=0).round(6).tolist(),
            "threshold": float(abs(deviation_threshold)),
            "units": deviation_map.units,
            "fit_rmse": deviation_map.fit_rmse,
        }
        dent_records.append(dent)
        insights.append(
            f"Detected {dent['severity']} dent {dent['dent_id']} with depth {depth:.4f} {deviation_map.units}."
        )
        insight_objects.append(
            build_insight(
                tool="detect_pipe_dents",
                title=f"Dent {dent['dent_id']}",
                message="Inward deformation cluster detected.",
                evidence=dent,
                severity="warning",
            )
        )

    diagnostics: list[str] = []
    if not dent_records:
        diagnostics.append("Candidate inward deviations did not form clusters above the minimum cluster size.")

    return build_result(
        insights=insights or ["No dents detected above the configured inward-deviation threshold."],
        insight_objects=insight_objects,
        diagnostics=diagnostics,
        data={"dents": dent_records},
        charts=[{"type": "heatmap", "title": "Detected dents"}],
    )



def measure_pipe_ovality(
    cloud: PointCloudData,
    pipe_fit: PipeFit,
    slice_spacing: float,
    units: str = "unitless",
) -> dict[str, Any]:
    axial = pipe_fit.axial_positions
    radii = pipe_fit.radial_distances
    if slice_spacing <= 0:
        return build_result(
            insights=["Ovality measurement skipped because slice spacing must be positive."],
            diagnostics=["slice_spacing must be greater than zero."],
        )

    start = float(axial.min())
    stop = float(axial.max())
    bins = np.arange(start, stop + slice_spacing, slice_spacing)
    if len(bins) < 2:
        bins = np.array([start, stop + slice_spacing], dtype=float)

    slices: list[dict[str, Any]] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (axial >= lower) & (axial < upper)
        if int(mask.sum()) < 8:
            continue
        slice_radii = radii[mask]
        max_radius = float(slice_radii.max())
        min_radius = float(slice_radii.min())
        ovality = float((max_radius - min_radius) / max(pipe_fit.radius, 1e-9))
        slices.append(
            {
                "axial_start": lower,
                "axial_end": upper,
                "max_radius": max_radius,
                "min_radius": min_radius,
                "ovality": ovality,
                "units": units,
            }
        )

    if not slices:
        return build_result(
            insights=["Ovality measurement skipped because slices did not contain enough points."],
            diagnostics=["Increase slice spacing or use denser point clouds for ovality analysis."],
        )

    max_slice = max(slices, key=lambda row: row["ovality"])
    return build_result(
        insights=[
            f"Maximum slice ovality is {max_slice['ovality']:.4f} between {max_slice['axial_start']:.4f} and {max_slice['axial_end']:.4f} {units}."
        ],
        insight_objects=[
            build_insight(
                tool="measure_pipe_ovality",
                title="Ovality summary",
                message="Cross-sectional ovality was measured along the pipe axis.",
                evidence={"max_ovality": max_slice, "slice_count": len(slices)},
            )
        ],
        data={"slices": slices, "max_ovality": max_slice},
        charts=[{"type": "line", "title": "Ovality profile"}],
    )



def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    grid = np.floor(points / voxel_size).astype(np.int64)
    unique_grid, inverse = np.unique(grid, axis=0, return_inverse=True)
    sums = np.zeros((len(unique_grid), 3), dtype=float)
    counts = np.bincount(inverse)
    for index in range(3):
        sums[:, index] = np.bincount(inverse, weights=points[:, index])
    return sums / counts[:, None]



def _inlier_mask(points: np.ndarray, neighbors: int, std_ratio: float) -> np.ndarray:
    n_neighbors = min(neighbors + 1, len(points))
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(points)
    distances, _ = model.kneighbors(points)
    mean_distance = distances[:, 1:].mean(axis=1)
    threshold = mean_distance.mean() + (mean_distance.std() * std_ratio)
    return mean_distance <= threshold



def _estimate_normals(points: np.ndarray, neighbor_count: int) -> np.ndarray:
    n_neighbors = min(neighbor_count + 1, len(points))
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(points)
    indices = model.kneighbors(points, return_distance=False)
    normals = np.zeros_like(points)
    centroid = points.mean(axis=0)
    for row_index, neighbors in enumerate(indices):
        neighborhood = points[neighbors[1:]]
        centered = neighborhood - neighborhood.mean(axis=0)
        covariance = centered.T @ centered / max(len(centered), 1)
        _, eigvecs = np.linalg.eigh(covariance)
        normal = eigvecs[:, 0]
        radial_hint = points[row_index] - centroid
        if np.dot(normal, radial_hint) < 0:
            normal = -normal
        normals[row_index] = normal / max(np.linalg.norm(normal), 1e-9)
    return normals



def _resolve_axis(points: np.ndarray, axis_hint: list[float] | tuple[float, float, float] | None) -> np.ndarray | None:
    if axis_hint is not None:
        axis = np.asarray(axis_hint, dtype=float)
        if axis.shape != (3,):
            return None
        norm = np.linalg.norm(axis)
        if norm <= 1e-9:
            return None
        return axis / norm

    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        return None
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    return axis / max(np.linalg.norm(axis), 1e-9)



def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_1 = np.cross(axis, reference)
    basis_1 = basis_1 / max(np.linalg.norm(basis_1), 1e-9)
    basis_2 = np.cross(axis, basis_1)
    basis_2 = basis_2 / max(np.linalg.norm(basis_2), 1e-9)
    return basis_1, basis_2



def _fit_circle_2d(points_2d: np.ndarray) -> tuple[np.ndarray, float]:
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    design = np.column_stack((2 * x, 2 * y, np.ones(len(points_2d))))
    rhs = x ** 2 + y ** 2
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    center = solution[:2]
    radius = float(np.sqrt(max(solution[2] + center[0] ** 2 + center[1] ** 2, 1e-12)))
    return center, radius



def _cylindrical_coordinates(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    basis_1: np.ndarray,
    basis_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centered = points - center
    axial_positions = centered @ axis
    radial_vectors = centered - np.outer(axial_positions, axis)
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    radial_x = radial_vectors @ basis_1
    radial_y = radial_vectors @ basis_2
    angles = np.arctan2(radial_y, radial_x)
    return axial_positions, radial_vectors, radial_distances, angles



def _cluster_radius(points: np.ndarray) -> float:
    if len(points) < 4:
        return 1.0
    n_neighbors = min(4, len(points))
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(points)
    distances, _ = model.kneighbors(points)
    neighbor_distance = distances[:, 1:].mean(axis=1)
    return float(max(np.median(neighbor_distance) * 2.5, 1e-6))



def _angular_span(angles: np.ndarray) -> float:
    normalized = np.mod(angles, 2 * np.pi)
    normalized.sort()
    if len(normalized) == 1:
        return 0.0
    gaps = np.diff(np.concatenate([normalized, normalized[:1] + (2 * np.pi)]))
    return float((2 * np.pi) - gaps.max())



def _severity_for_depth(depth: float, radius: float, severity_bands: dict[str, float]) -> str:
    ratio = depth / max(radius, 1e-9)
    ordered = sorted(severity_bands.items(), key=lambda item: item[1])
    severity = ordered[0][0]
    for label, threshold in ordered:
        severity = label
        if ratio < threshold:
            break
    return severity
