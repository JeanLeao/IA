# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score, accuracy_score


# # Carrega o arquivo CSV em um DataFrame
# data = pd.read_csv('allgames.csv')

# # Divide os dados em duas partes
# train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
# pd.set_option('display.max_columns', None)
# train_data = train_data.replace('K', '00', regex=True)

# # Substitui todas as ocorrências de "K" por "00" no DataFrame de teste
# test_data = test_data.replace('K', '00', regex=True)

# # Definir intervalos para classificação
# bins = [0, 3, 4, 5]  # Defina os intervalos de classificação desejados
# labels = ['Baixa', 'Média', 'Alta']  # Defina as classes correspondentes aos intervalos

# # Converter a coluna 'Rating' em categorias com base nos intervalos
# train_data['Rating'] = pd.cut(train_data['Rating'], bins=bins, labels=labels)
# test_data['Rating'] = pd.cut(test_data['Rating'], bins=bins, labels=labels)

# # Selecionar as colunas relevantes para classificação
# features = ['Title', 'Release Date', 'Team', 'Times Listed', 'Number of Reviews', 'Genres', 'Plays', 'Playing', 'Backlogs', 'Wishlist']
# target = 'Rating'

# # Remover linhas com valores NaN dos dados de treinamento
# train_data = train_data.dropna(subset=[target])

# train_X = train_data[features]
# train_y = train_data[target]
# test_X = test_data[features]

# # Transformar colunas categóricas em variáveis dummy separadamente para o conjunto de treinamento e teste
# train_X = pd.get_dummies(train_X, columns=['Title', 'Release Date', 'Team', 'Genres'])
# test_X = pd.get_dummies(test_X, columns=['Title', 'Release Date', 'Team', 'Genres'])

# # Alinhar as colunas de recursos nos conjuntos de treinamento e teste
# train_X, test_X = train_X.align(test_X, join='outer', axis=1, fill_value=0)

# # Treinar o modelo
# model = RandomForestClassifier()
# model.fit(train_X, train_y)

# # Classificar os registros de teste
# predictions = model.predict(test_X)

# # Adicionar as classificações previstas ao DataFrame de teste
# test_data['Predicted Rating'] = predictions



# # Exibe as predições
# print("Predições:")
# print(test_data[['Title', 'Predicted Rating']].head())

# test_data['Rating'] = test_data['Rating'].astype(str)

# # Calcular as métricas de desempenho

# # ...

# # Calcular as métricas de desempenho
# precision = precision_score(test_data['Rating'], predictions, average='weighted', zero_division=1)
# recall = recall_score(test_data['Rating'], predictions, average='weighted')
# accuracy = accuracy_score(test_data['Rating'], predictions)

# # Exibir as métricas de desempenho
# print("Precision:", precision)
# print("Recall:", recall)
# print("Accuracy:", accuracy)

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Carrega o arquivo CSV em um DataFrame
data = pd.read_csv('allgames.csv')

# Divide os dados em duas partes
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
pd.set_option('display.max_columns', None)
train_data = train_data.replace('K', '00', regex=True)

# Substitui todas as ocorrências de "K" por "00" no DataFrame de teste
test_data = test_data.replace('K', '00', regex=True)

# Definir intervalos para classificação
bins = [0, 3, 4, 5]  # Defina os intervalos de classificação desejados
labels = ['Baixa', 'Média', 'Alta']  # Defina as classes correspondentes aos intervalos

# Converter a coluna 'Rating' em categorias com base nos intervalos
train_data['Rating'] = pd.cut(train_data['Rating'], bins=bins, labels=labels)
test_data['Rating'] = pd.cut(test_data['Rating'], bins=bins, labels=labels)

# Selecionar as colunas relevantes para classificação
features = ['Title', 'Release Date', 'Team', 'Times Listed', 'Number of Reviews', 'Genres', 'Plays', 'Playing', 'Backlogs', 'Wishlist']
target = 'Rating'

# Remover linhas com valores NaN dos dados de treinamento
train_data = train_data.dropna(subset=[target])

train_X = train_data[features]
train_y = train_data[target]
test_X = test_data[features]

# Transformar colunas categóricas em variáveis dummy separadamente para o conjunto de treinamento e teste
train_X = pd.get_dummies(train_X, columns=['Title', 'Release Date', 'Team', 'Genres'])
test_X = pd.get_dummies(test_X, columns=['Title', 'Release Date', 'Team', 'Genres'])

# Alinhar as colunas de recursos nos conjuntos de treinamento e teste
train_X, test_X = train_X.align(test_X, join='outer', axis=1, fill_value=0)

# Padronizar os dados
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# Criar a rede neural
model = MLPClassifier(hidden_layer_sizes=(100, 50), learning_rate_init=0.001, activation='relu')

# Treinar o modelo
model.fit(train_X, train_y)

# Classificar os registros de teste
predictions = model.predict(test_X)

# Adicionar as classificações previstas ao DataFrame de teste
test_data['Predicted Rating'] = predictions

# Exibe as predições
print("Predições:")
pd.set_option('display.max_rows', None)
print(test_data[['Title', 'Predicted Rating']])

test_data['Rating'] = test_data['Rating'].astype(str)

# Calcular as métricas de desempenho
precision = precision_score(test_data['Rating'], predictions, average='weighted', zero_division=1)
recall = recall_score(test_data['Rating'], predictions, average='weighted')
accuracy = accuracy_score(test_data['Rating'], predictions)

# Exibir as métricas de desempenho
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
