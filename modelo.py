from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

def modelo(x,y, persona):
    x_ent,x_pru,y_ent,y_pru = train_test_split(x,y,test_size=0.2)

    modelo = LogisticRegression(max_iter=10000)
    modelo.fit(x_ent,y_ent)


    predicciones = modelo.predict(x_pru)

    accuracy_score(y_pru,predicciones)


    confusion_matrix(y_pru,predicciones)

    pd.DataFrame(confusion_matrix(y_pru,predicciones),
                columns =['Pred:No','Pred;Si'],index=['Real:No','Real:Si'])
    
    if(modelo.predict([persona])==0):
        return "NO tiene ANEMIA"
    else:
        return "SI tiene ANEMIA"