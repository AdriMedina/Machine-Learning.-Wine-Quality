
library(leaps)
library(cvTools)
library(MASS)

# Cargamos las tablas de los ficheros.
# WineQualityRed <- read.csv("C:/Users/Adri/Documents/GitHub/Machine-Learning.-Wine-Quality/winequality-red.csv", sep=";")
# WineQualityWhite <- read.csv("C:/Users/Adri/Documents/GitHub/Machine-Learning.-Wine-Quality/winequality-white.csv", sep=";")


######
# Alternativa. Función que devuelve el data frame correspondiente a NombreArchivo.
#
# Funcionamiento:
###
### Se asume que el directorio de trabajo de R está situado en la carpeta "~/Documents" (Carpeta 'Mis documentos')
### Luego se añade la ruta "/GitHub/Machine-Learning.-Wine-Quality/" a "~/Documents" por lo que se obtiene:
###
###     "~/Documents/GitHub/Machine-Learning.-Wine-Quality/"
###
### A continuacion se añade a esa ruta el nombre del fichero pasado como argumento y se lee finalemente
### la dirección:
###
###     "~/Documents/GitHub/Machine-Learning.-Wine-Quality/<NombreArchivo>"
######


#LeerFichero<-function(NombreArchivo){
#  direccion <- paste(getwd(), "/GitHub/Machine-Learning.-Wine-Quality/", NombreArchivo, sep="")
#  return(read.csv(direccion, sep=";"))  
#}
LeerFichero<-function(NombreArchivo){
  direccion <- paste(getwd(), "/", NombreArchivo, sep="")
  return(read.csv(direccion, sep=";"))  
}



# Llamamos a la función "LeerFichero".
WineQualityRed <- LeerFichero("winequality-red.csv")
WineQualityWhite <- LeerFichero("winequality-white.csv")

# Equilibramos los datos replicando de los que menos hay hasta igualarlos, es decir, balancearlos
# Ajustamos una función para hacerlo según el dataframe que le pasemos (vino blanco o tinto)
balanceDataFrame <- function (dataset){
  balanceAux <- dataset
  contador = 1
  set.seed(1)
  
  for(i in min(balanceAux$quality):max(balanceAux$quality)){
    # Guardamos los indices de las filas en un vector que tienen calidad = i
    index_iQuality <- which(balanceAux$quality == i)
    
    # Hacemos boostraping del subconjunto de calidad i-esima, para obtener el mismo número de filas que el máximo.
    index_iQuality.sample <- sample(index_iQuality, size=(max(table(balanceAux$quality)) - table(balanceAux$quality)[contador]), replace=TRUE)
    contador = contador + 1
    
    # Añadimos al dataframe RedEquilibrado las filas correspondientes al index_iQuality.sample
    dataset <- rbind(dataset, balanceAux[index_iQuality.sample, ])
  }
  return(dataset)
}


# Aplicamos la función a nuestros dos dataframe
BalanceWineRed <- balanceDataFrame(WineQualityRed)
BalanceWineWhite <- balanceDataFrame(WineQualityWhite)


# Mezclamos los datos del dataframe para asegurarnos que funciona la cross-validation
rand <- sample(nrow(BalanceWineRed))
BalanceWineRed <- BalanceWineRed[rand,]
rand <- sample(nrow(BalanceWineWhite))
BalanceWineWhite <- BalanceWineWhite[rand,]




##############################
### Selección de variables ###
##############################

# Funcion que realiza predicciones usando el mejor subconjunto de variables calculado.
predict.regsubsets <- function(object, newdata, id, ...)
{
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coeficientes <- coef(object, id=id)
  xvars <- names(coeficientes)
  mat[,xvars]%*%coeficientes
}

# Usando validación cruzada con k-iteraciones, aplicamos la técnica de "Selección del mejor subconjunto" para cada uno de los
# k-conjuntos de entrenamiento.
k <- 10
set.seed(1)

# Definimos los conjuntos de validación
Red.folds <- sample(1:k, nrow(BalanceWineRed), replace=TRUE)
White.folds <- sample(1:k, nrow(BalanceWineWhite), replace=TRUE)

# Definimos la matriz que contendrá los datos.
Red.cv.errors <- matrix(NA, k, 11, dimnames=list(NULL, paste(1:11)))
White.cv.errors <- matrix(NA, k, 11, dimnames=list(NULL, paste(1:11)))

# En cada iteracion J, los elementos de 'folds == j' corresponden al conjunto de test.
# y el resto de elementos al conjunto de entrenamiento.
for(j in 1:k){
  # Obtenemos el mejor subconjunto de variables para cada conjunto de entrenamiento.
  Red.best_fit <- regsubsets(quality~.,data=BalanceWineRed[Red.folds!=j,], nvmax=11)
  White.best_fit <- regsubsets(quality~.,data=BalanceWineWhite[White.folds!=j,], nvmax=11)
  
  # Para cada uno de los tamaños de problema posibles
  for(i in 1:11){
    # Realizamos predicciones sobre la muestra de test usando las mejores variables escogidas
    # en la iteracion J.
    Red.pred <- predict(Red.best_fit, BalanceWineRed[Red.folds==j,], id=i)
    White.pred <- predict(White.best_fit, BalanceWineWhite[White.folds==j,], id=i)
    
    # Calculamos el MSE para cada uno de los tamaños (1:11) y los almacenamos en la matriz
    # cv.errors en la fila correspondiente a la iteración J-esima.
    Red.cv.errors[j,i] <- mean( (BalanceWineRed$quality[Red.folds==j]-Red.pred)^2 )
    White.cv.errors[j,i] <- mean( (BalanceWineWhite$quality[White.folds==j]-White.pred)^2 )
  }
}

# Hacemos la media por columnas de todos los datos de la matriz para obtener la media
# del error en todas la iteraciones de la validación cruzada.
Red.mean.cv.errors <- apply(Red.cv.errors, 2, mean)
White.mean.cv.errors <- apply(White.cv.errors, 2, mean)

# Pintamos el error en función del número de variables obtenido por validación cruzada.
par(mfrow=c(1,2))
plot(Red.mean.cv.errors, type='b')
plot(White.mean.cv.errors, type='b')

# Escogemos el tamaño de modelo que ha obtenido el menor error cuadrático medio MSE.
Red.best_size <- which.min(Red.mean.cv.errors)
White.best_size <- which.min(White.mean.cv.errors)

# Obtenemos los coeficientes de ese tamaño de modelo usando esta vez la muestra completa
# y no solo la de training.
Red.best_variables <- regsubsets(quality~., data=BalanceWineRed, nvmax=11)
White.best_variables <- regsubsets(quality~., data=BalanceWineWhite, nvmax=11)

# Y extraemos los coeficientes que obtuvieron menor MSE en la muestra de training
Red.best_variables.coeficientes <- coef(Red.best_variables, id=Red.best_size)
White.best_variables.coeficientes <- coef(White.best_variables, id=White.best_size)



# Guardamos las formulas con las variables seleccionadas con CV
Red.form <- as.formula(quality~fixed.acidity+volatile.acidity+residual.sugar+chlorides+pH+sulphates+alcohol)
White.form <- as.formula(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+density+pH+sulphates)


# Dividimos los conjuntos en training y test
train_index <- sample(seq_len(nrow(BalanceWineRed)), size = floor(0.7*nrow(BalanceWineRed)))
Red.training <- BalanceWineRed[train_index, ]
Red.test <- BalanceWineRed[-train_index, ]

train_index <- sample(seq_len(nrow(BalanceWineWhite)), size = floor(0.7*nrow(BalanceWineWhite)))
White.training <- BalanceWineWhite[train_index, ]
White.test <- BalanceWineWhite[-train_index, ]



# Vamos a empezar ajustando modelos LDA, QDA, KNN con cross-validation
# LDA
cv_lda <- function(dataframe, formula){
  # Creamos los folds para el conjunto de datos, es decir, el reparto para CV
  k <- 10
  folds <- cvFolds(nrow(dataframe), k, R=1)
  
  # Creamos un vector para guardar los errores de CV para hacer la media después
  cvError <- matrix(NA, 1, 10)
  
  for(i in 1:k){
    # Asignamos los conjuntos de training y validacion
    validation_set <- dataframe[folds$subsets[folds$which==i, ], ]
    training_set <- dataframe[folds$subsets[folds$which!=i, ], ]
    
    # Ajustamos el modelo LDA y predecimos sobre el conjunto de validacion
    lda.fit = lda(formula, data=training_set)
    lda.pred = predict(lda.fit, validation_set)
    
    # Creamos la matriz de confusión y calculamos el error. Después lo guardamos
    lda.class = lda.pred$class
    MC <- table(lda.class, validation_set$quality)
    cvError[1, i] = 1 - mean(lda.class==validation_set$quality)
  }
  # Devolvemos la media de los errores acumulados
  return(mean(cvError))
}


print(cv_lda(Red.training, Red.form))



