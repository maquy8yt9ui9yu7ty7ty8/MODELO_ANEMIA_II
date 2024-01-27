import modelo as mdl
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
import io
from flask import send_file
import os

app = Flask(__name__)


# Importar datos
datos = pd.read_csv("kaggle/input/anemia.csv")
x = datos.drop(['Result'], axis=1)
y = datos['Result']

def obtener_primeros_15(datos):
    primeros_15 = datos.head(15)
    return primeros_15

def descargar_csv(datos):
    csv_output = io.StringIO()
    datos.to_csv(csv_output, index=False)
    response = send_file(
        io.BytesIO(csv_output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='datos_anemia.csv'
    )

    return response

########################################################################################################################################
#RUTAS PARA INICIAR LA APLICACION
#RUTA PRINCIPAL
@app.route("/")
def prin():
    return render_template("index.html")
#RUTA QUE NOS LLEVA A PEDIR DATOS PARA PREDECIR 
@app.route("/predecir_anemia")
def predecir_anemia():
    return render_template("ingresar_datos.html")

# ruta que procesa los datos que pedimos para predecir
@app.route("/procesar_datos", methods=['POST'])
def procesar_datos():
    if request.method == 'POST':
        dato1 = int(request.form['dato1'])
        dato2 = float(request.form['dato2'])
        dato3 = float(request.form['dato3'])
        dato4 = float(request.form['dato4'])
        dato5 = float(request.form['dato5'])

        # Puedes devolver una respuesta a la página
        resul = mdl.modelo(x,y,[dato1, dato2, dato3, dato4, dato5])
        if dato1 == 0:
            dato1 = 'Masculino'
        else:
            dato1 = 'Femenino'
        return render_template("procesar_datos.html", dato1=dato1, dato2=dato2, dato3=dato3, dato4=dato4, dato5=dato5, resul=resul)


#RUTA MOSTRAR TABLA DE REPORTE
@app.route("/mostrar_tabla")
def mostrar_tabla():
    data = obtener_primeros_15(datos)   
    return render_template('reporte_tabla.html',data=data)

@app.route("/descargar_tabla")
def descargar_tabla():
    return descargar_csv(datos)


if __name__ == "__main__":
    app.run(debug=True)
