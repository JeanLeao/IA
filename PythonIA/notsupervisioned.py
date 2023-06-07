import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mplcursors

# Carrega o arquivo CSV em um DataFrame
data = pd.read_csv('allgames.csv')

# Selecionar as colunas relevantes para análise
features = ['Title', 'Release Date', 'Team', 'Times Listed', 'Number of Reviews', 'Genres', 'Plays', 'Playing', 'Backlogs', 'Wishlist']

# Filtrar os dados relevantes
data_filtered = data[features]

# Substituir todas as ocorrências de "K" por "00" no DataFrame
data_filtered = data_filtered.replace('K', '00', regex=True)

# Preencher valores ausentes com zeros
data_filtered = data_filtered.fillna(0)

# Aplicar one-hot encoding nas colunas categóricas
data_encoded = pd.get_dummies(data_filtered, columns=['Title', 'Release Date', 'Team', 'Genres'])

# Padronizar os dados
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Aplicar PCA para redução de dimensionalidade
pca = PCA(n_components=2)
components = pca.fit_transform(data_scaled)
X_pca = components[:, 0:2]

# Realizar clustering com K-means nos componentes principais
kmeans = KMeans(n_clusters=20)  # Definir o número de clusters desejado
clusters = kmeans.fit_predict(X_pca)

# Adicionar as informações de cluster ao DataFrame original
data['Cluster'] = clusters

# Agrupar os jogos por cluster e calcular a média do rating para cada grupo
grouped_data = data.groupby('Cluster')['Rating'].mean()

# Exibir os resultados do agrupamento
print(grouped_data)

# Visualizar os clusters com rótulos de cada jogo
for cluster in range(20):  # Altere o número 3 para o número de clusters definido anteriormente
    cluster_games = data[data['Cluster'] == cluster][['Title', 'Genres']]
    print(f"Cluster {cluster}:")
    print(cluster_games)
    print()

# Visualizar os clusters nos componentes principais com rótulos de cada jogo
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')

# Adicionar rótulos aos pontos no gráfico
labels = [f"{title}\nGenre: {genres}" for title, genres in zip(data['Title'], data['Genres'])]
cursor = mplcursors.cursor(scatter, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_title('Clustering com PCA')
plt.show()