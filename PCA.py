import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import LabelEncoder


class PCA:
    def __init__(self, data):
        self.original_data = data
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.numeric_data = None
        self.mean = None
        self.components = None
        self.transformed_data = None
        self.centered_data = None
        self.cov_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, n_components):
        if n_components > self.data.shape[1]:
            raise ValueError("n_components cannot exceed the number of features in the data.")
        self.mean = np.mean(self.numeric_data, axis=0)
        self.centered_data = self.numeric_data - self.mean

        self.cov_matrix = np.cov(self.centered_data, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov_matrix)
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        self.components = self.eigenvectors[:, sorted_indices[:n_components]]

    def transform(self):
        if self.mean is None or self.components is None:
            raise ValueError("The PCA model has not been fitted yet.")
        self.transformed_data = np.dot(self.centered_data, self.components)
        return self.transformed_data

    def plot_data(self):

        if self.numeric_data.shape[1] == 2:
            plt.figure(figsize=(16, 10))
            gs = GridSpec(1, 3, width_ratios=[3, 0.5, 3])


            ax1 = plt.subplot(gs[0])
            ax1.scatter(self.numeric_data[:, 0], self.numeric_data[:, 1], color="black", label="Original Data")
            ax1.set_title("Original Data")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.grid(True)

            ax3 = plt.subplot(gs[2])
            ax3.scatter(self.centered_data[:, 0], self.centered_data[:, 1], color="black", label="Centered Data")
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y")
            ax3.set_title("Centered Data with Bidirectional Eigenvector Axes")

            scale_factor = np.max(np.linalg.norm(self.centered_data, axis=1))
            origin = np.array([0, 0])

            for i in range(len(self.eigenvalues)):
                color = "blue" if i == 0 else "orange"
                label = "PC #1" if i == 0 else "PC #2"

                ax3.quiver(
                    origin[0], origin[1],
                    self.eigenvectors[0, i] * scale_factor,
                    self.eigenvectors[1, i] * scale_factor,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=color,
                    label=label
                )
                ax3.quiver(
                    origin[0], origin[1],
                    -self.eigenvectors[0, i] * scale_factor,
                    -self.eigenvectors[1, i] * scale_factor,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=color
                )

            ax3.grid(True)
            ax3.axis("equal")
            ax3.legend()
            plt.tight_layout()
            plt.show()

        elif self.numeric_data.shape[1] > 2:

            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(
                self.centered_data[:, 0],
                self.centered_data[:, 1],
                self.centered_data[:, 2],
                color='blue',
                label='Centered Data',
                alpha=0.7
            )


            scale_factor = np.max(np.linalg.norm(self.centered_data, axis=1))
            for i in range(self.components.shape[1]):
                ax.quiver(0, 0, 0,
                          self.eigenvectors[0, i] * scale_factor,
                          self.eigenvectors[1, i] * scale_factor,
                          self.eigenvectors[2, i] * scale_factor,
                          color='r' if i == 0 else 'g',
                          label=f"Eigenvector {i + 1}")

                ax.quiver(0, 0, 0,
                          -self.eigenvectors[0, i] * scale_factor,
                          -self.eigenvectors[1, i] * scale_factor,
                          -self.eigenvectors[2, i] * scale_factor,
                          color='r' if i == 0 else 'g')

            ax.set_title("3D Visualization of Centered Data with Eigenvectors")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()
            plt.show()

    def plot_covariance(self):
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix has not been computed. Fit the model first.")

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            self.cov_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            cbar=True
        )
        plt.title("Covariance Matrix")
        plt.show()

    def loss_calc(self, n_components):
        if self.eigenvalues is None:
            raise ValueError("The PCA model has not been fitted yet.")

        total_variance = np.sum(self.eigenvalues)
        explained_variance = np.sum(self.eigenvalues[:n_components])
        information_loss = 1 - (explained_variance / total_variance)

        print(f"Total variance: {total_variance}")
        print(f"Explained variance (by selected components): {explained_variance}")
        print(f"Information loss: {information_loss * 100:.2f}%")
        return information_loss

    def write_results(self, filename="PCA_Results.csv"):

        with open(filename, "w") as f:
            f.write("Original Data:\n")
            f.write(self.original_data.to_string() if isinstance(self.original_data, pd.DataFrame) else str(
                self.original_data))
            f.write("\n\nCentered Data:\n")
            f.write(pd.DataFrame(self.centered_data).to_string())
            f.write("\n\nCovariance Matrix:\n")
            f.write(pd.DataFrame(self.cov_matrix).to_string())
            f.write("\n\nEigenvalues:\n")
            f.write(np.array2string(self.eigenvalues, precision=4))
            f.write("\n\nEigenvectors:\n")
            f.write(pd.DataFrame(self.eigenvectors).to_string())
            f.write("\n\nReduced Data:\n")
            f.write(pd.DataFrame(self.transformed_data).to_string())
            information_loss = self.loss_calc(len(self.components.T))
            f.write("\n\nInformation Loss:\n")
            f.write(f"{information_loss * 100:.2f}%\n")
            f.write(20 * "-")

    def run_all(self, n_components, output_file="PCA_Result.txt"):
        if self.data.select_dtypes(include=['object']).shape[1] > 0:
            label_encoder = LabelEncoder()
            for column in self.data.select_dtypes(include=['object']).columns:
                self.data[column] = label_encoder.fit_transform(self.data[column])

        self.numeric_data = self.data.select_dtypes(include=[np.number]).to_numpy()
        self.fit(n_components)
        self.transform()
        self.plot_data()
        self.plot_covariance()
        self.loss_calc(n_components)
        self.write_results(output_file)