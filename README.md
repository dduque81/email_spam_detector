# Detector de Spam de emails
Repositorio desarrollado con fines practicos para desarrollar habilidades de NLP
## Descripción del dataset
El dataset consta de textos recibidos en un email, y su respectiva clasificación como spam o no spam, los textos se encuentran en idioma ingles.
## Metodología
##### Limpieza y preparación de datos
Para esta sección se remueven etiquetas html, corchetes, caracteres especiales, emails, numeros, todos los textos se dejan en minuscula, remoción de stopwords, lematización y tokenización.
##### Exploración de los datos
###### Distribución
![image](https://github.com/dduque81/email_spam_detector/assets/103476375/ee4b0c2b-bc8a-4c7a-968a-c73cde583282)
###### Bigramas y trigramas
![image](https://github.com/dduque81/email_spam_detector/assets/103476375/a332a77e-aa81-4688-a683-5181b1018770)
###### Nubes de palabras
![image](https://github.com/dduque81/email_spam_detector/assets/103476375/e4421955-50fd-471f-93ea-cc7c2d29d6f9)

Finalmente se realizan modelos de machine learning: regresión logistica, arboles de decisión, y Naive Bayes

