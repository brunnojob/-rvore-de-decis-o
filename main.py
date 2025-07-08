import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# var

# ler csv
df = pd.read_csv("pecas.csv")

# transformar dados categoricos em numericos
le = LabelEncoder()
df["tipo_peca"] = le.fit_transform(df["tipo_peca"])

# remove 1 coluna dos itens citados
X = df.drop(["tipo_peca", "n_serie", "coef_expansao"], axis=1)
y = df["tipo_peca"]

# padronizar os dados numericos (colunas na mesma escala.)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# dividir os dados em 2 grupos: treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# funções criadas ( Modelo que treina e testa árvore de decisão e svm, desempenho árvore de decisão e svm, classificação de nova peça e desenho da árvore de decisão.)
def modelo_arvore(X_train, y_train):
    clf_tree = DecisionTreeClassifier()
    clf_tree.fit(X_train, y_train)
    return clf_tree

def modelo_svm(X_train, y_train):
    clf_svm = SVC(kernel="rbf", C=1.0)
    clf_svm.fit(X_train, y_train)
    return clf_svm


def desempenho_arvore_decisao(modelo, X_test, y_test, le): #como o senhor pediu, tentei deixar a função de mostrar o desempenho aqui e quando rodar para mostrar o desempenho somente utilizar o nome da função.
    y_pred = modelo.predict(X_test)
    print("\nDesempenho da Árvore de Decisão")
    print("Desempenho:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

def desempenho_svm(modelo, X_test, y_test, le): #como o senhor pediu, tentei deixar a função de mostrar o desempenho aqui e quando rodar para mostrar o desempenho somente utilizar o nome da função.
    y_pred = modelo.predict(X_test)
    print("\nDesempenho da SVM")
    print("Desempenho:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))


def classificar_nova_peca(modelo): # função para classificar uma nova peça dos 2 modelos
    try:
        print("\nDigite os dados da nova peça para classificação:")
        espessura = float(input("Espessura (mm): "))
        densidade = float(input("Densidade (g/cm³): "))
        tolerancia_dim = float(input("Tolerância dimensional (mm): "))
        temperatura_fusao = float(input("Temperatura de fusão (°C): "))
        nova_entrada = [[espessura, densidade, tolerancia_dim, temperatura_fusao]]
        entrada_escalada = scaler.transform(nova_entrada) #nova entrada
        predicao = modelo.predict(entrada_escalada)
        print("Peça classificada como:", le.inverse_transform(predicao)[0])
    except Exception as e:
        print("Erro ao classificar nova peça:", e)


def desenho_arvore(modelo, X, le):
    plt.figure(figsize=(16, 8))
    plot_tree(modelo, feature_names=X.columns, class_names=le.classes_, filled=True)
    plt.title("Árvore de Decisão")
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#começo do código
while True: #interface menu
    print("\nEscolha o modelo:")
    print("1 - Árvore de Decisão")
    print("2 - SVM")
    print("3 - Sair")
    escolha = input("> ")

    if escolha == "1":
        clf_tree = modelo_arvore(X_train, y_train) # criar e treinar um modelo de arvore de decisao

        while True:
            print("\nModelo: Árvore de Decisão")
            print("1 - Ver desempenho")
            print("2 - Classificar nova peça")
            print("3 - Desenhar árvore")
            print("4 - Voltar")
            sub = input("> ")

            if sub == "1": # mostrar o desempenho
               desempenho_arvore_decisao(clf_tree, X_test, y_test, le) # somente o nome da função para evitar repetir código como o senhor pediu.
            elif sub == "2": # classificar uma nova entrada de peça
                classificar_nova_peca(clf_tree)
            elif sub == "3": # irá mostrar o desenho da arvore
                desenho_arvore(clf_tree, X, le)  # somente o nome da função para evitar repetir código como o senhor pediu.
            elif sub == "4": # voltar
                break
            else:
                print("Opção inválida. Tente novamente.")

    elif escolha == "2": # criar e treinar um modelo de svm
        clf_svm = modelo_svm(X_train, y_train) # somente o nome da função para evitar repetir código como o senhor pediu.

        while True:
            print("\nModelo: SVM")
            print("1 - Ver desempenho")
            print("2 - Classificar nova peça")
            print("3 - Voltar")
            sub = input("> ")

            if sub == "1": # mostrar desempenho do svm
              desempenho_svm(clf_svm, X_test, y_test, le) # somente o nome da função para evitar repetir código como o senhor pediu.
            elif sub == "2": # classificar nova peça
                classificar_nova_peca(clf_svm)
            elif sub == "3": # voltar
                break
            else:
                print("Opção inválida. Tente novamente.")

    elif escolha == "3": # encerra o codigo
        print("Programa encerrado.")
        break
    else:
        print("Opção inválida. Tente novamente.")
