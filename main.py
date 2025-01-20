from PCA import PCA
import pandas as pd

if __name__ == '__main__':
    file_path = input("Enter the CSV file path (e.g., data.csv): ")
    n_components = int(input("Enter the number of components (e.g., 2): "))
    try:
        data = pd.read_csv(file_path)
        print(f"Data successfully loaded: {data.shape[0]} rows, {data.shape[1]} columns.")
    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        exit()

    pca = PCA(data.values)
    pca.run_all(
        n_components=n_components,
        output_file="PCA_Result.txt"
    )
