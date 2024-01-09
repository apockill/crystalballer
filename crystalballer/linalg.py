import numpy as np
import numpy.typing as npt


def normalize_vector(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize a vector and handle the zero vector case."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector")
    return vector / norm


def rotation_between_vectors(
    vec_a: npt.NDArray[np.float64], vec_b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculate the rotation matrix to rotate vector a to vector b."""

    a = normalize_vector(vec_a)
    b = normalize_vector(vec_b)
    cross_prod = np.cross(a, b)
    dot_prod = np.dot(a, b)

    if np.allclose(cross_prod, np.zeros(3)) and np.isclose(dot_prod, 1):
        # Vectors are parallel
        return np.eye(3)

    k_mat = np.array(
        [
            [0, -cross_prod[2], cross_prod[1]],
            [cross_prod[2], 0, -cross_prod[0]],
            [-cross_prod[1], cross_prod[0], 0],
        ]
    )

    rotation_matrix = (
        np.eye(3)
        + k_mat
        + k_mat.dot(k_mat) * (1 - dot_prod) / np.linalg.norm(cross_prod) ** 2
    )
    return rotation_matrix  # type: ignore
