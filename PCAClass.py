import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean = None
        self.cov_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.variance = None
        self.scores = None
        self.loadings = None

    # fit data to PC model 
    def fit(self, data):
        # Step 1: Mean-centering
        self.mean = np.mean(data, axis=0)
        centered_data = data - self.mean

        # Step 2: Calculate the covariance matrix
        self.cov_matrix = np.cov(centered_data, rowvar=False)

        # Step 3: Eigen-decomposition with np.linalg.eig()
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov_matrix)

        # Step 4: Project data onto the PC1/PC2 axis
        if self.n_components is not None:
            # Sort lambda and eigenvectors in descending order
            sorted_indices = np.argsort(self.eigenvalues)[::-1]
            self.eigenvalues = self.eigenvalues[sorted_indices][:self.n_components]
            self.eigenvectors = self.eigenvectors[:, sorted_indices][:, :self.n_components]

        # Use dot product to find scores
        self.scores = np.dot(centered_data, self.eigenvectors)

        # Calculate total variance and percent variance that is explained by 1st 2 PC's
        total_variance = np.sum(self.eigenvalues)
        self.variance = self.eigenvalues / total_variance * 100

        # Calculate loadings
        self.loadings = self.eigenvectors * np.sqrt(self.eigenvalues)

    def transform(self, data):
        # Mean-center the data
        centered_data = data - self.mean

        # Project the data onto the principal component axes
        scores = np.dot(centered_data, self.eigenvectors)

        return scores

    def scores_plot(self):
        
        plt.figure(figsize=(8, 6))
        plt.scatter(self.scores[:, 0], self.scores[:, 1], c='b', marker='o', edgecolors='k')
        plt.title('Scores Plot')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()

    def scree_plot(self):

        plt.figure(figsize=(8, 6))
        plt.scatter(range(1, len(self.variance) + 1), self.variance, c='b', marker='o')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance (%)')
        plt.title('Scree Plot')
        plt.grid(True)
        plt.show()

    def data_projection(self, data):
        # Project raw data onto PC1 and PC2
        scores_pc1 = np.dot(data, pca.eigenvectors[:, 0])
        scores_pc2 = np.dot(data, pca.eigenvectors[:, 1])

        # Visualize the projected data using PCA class
        plt.figure(figsize=(8, 6))
        plt.scatter(scores_pc1, scores_pc2, c='b', marker='o', edgecolors='k')
        plt.title('Projected Data onto PC1 vs. PC2')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()

    def loadings_plot(self):
        # Create Loadings Plot
        plt.figure(figsize=(8, 6))
        pc1_loadings = self.loadings[:, 0]
        pc2_loadings = self.loadings[:, 1]

        # Create a scatter plot to visualize variable contributions to PC1 and PC2
        plt.scatter(pc1_loadings, pc2_loadings, marker='o', edgecolors='k')

        # Label points with variable names (assuming data.columns contains variable names)
        if hasattr(self, 'data'):
            for i, variable in enumerate(self.data.columns):
                plt.annotate(variable, (pc1_loadings[i], pc2_loadings[i]))
        
        plt.xlabel('PC1 Loadings')
        plt.ylabel('PC2 Loadings')
        plt.title('Loadings Plot (PC1 vs. PC2)')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
   # Load in CSV file
    csv_file = 'Homework_2_dataset_prob4.csv'
    data = pd.read_csv(csv_file, index_col=0)

    # Transpose the data
    data = data.values.T

    # Calculate the number of principal components based on the input data
    n_components = min(data.shape[0], data.shape[1])

    # Create an instance of the PCA class
    pca = PCA(n_components=n_components)

    # Fit the data using the PCA instance
    pca.fit(data)

    # Print results
    print(f"Variance of PC1 + PC2: {np.sum(pca.variance)}\nPC Scores: \n{pca.scores}\nLoadings: \n{pca.loadings}")

    pca.scores_plot()
    pca.data_projection(data)
    pca.scree_plot()
    pca.loadings_plot()