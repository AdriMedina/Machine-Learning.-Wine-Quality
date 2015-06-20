
library(leaps)

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
LeerFichero<-function(NombreArchivo){
  direccion <- paste(getwd(), "/GitHub/Machine-Learning.-Wine-Quality/", NombreArchivo, sep="")
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


# Dividimos los conjuntos en training y test
train_index <- sample(seq_len(nrow(BalanceWineRed)), size = floor(0.7*nrow(BalanceWineRed)))
Red.training <- BalanceWineRed[train_index, ]
Red.test <- BalanceWineRed[-train_index, ]

train_index <- sample(seq_len(nrow(BalanceWineWhite)), size = floor(0.7*nrow(BalanceWineWhite)))
White.training <- BalanceWineWhite[train_index, ]
White.test <- BalanceWineWhite[-train_index, ]


##############################
### Selección de variables ###
##############################

# Obtenemos el mejor conjunto de variables predictoras con "quality" como variable respuesta
# usando para ello la muestra de training.
Red.best_variables <- regsubsets(quality~., data=Red.training, nvmax=11)
White.best_variables <- regsubsets(quality~., data=White.training, nvmax=11)

# Calculamos el error del conjunto de validación para el mejor modelo de cada tamaño.
# Para ello primero creamos una 'model matrix' a partir de la muestra de test.
Red.test.model_matrix <- model.matrix(quality~., data=Red.test)
White.test.model_matrix <- model.matrix(quality~., data=White.test)

# Para cada uno de los tamaños de modelo (1-11) extraemos los coeficientes del conjunto "best_variables"
# calculado antes y los usamos para realizar predicciones con el objetivo de obtener el mejor MSE.
Red.MSE <- rep(NA,11)
White.MSE <- rep(NA,11)

for(i in 1:11){
  Red.coeficientes_tam_iesimo <- coef(Red.best_variables, id=i)
  White.coeficientes_tam_iesimo <- coef(White.best_variables, id=i)
  
  # Calculamos el MSE para los coeficientes del modelo de tamaño i-esimo.
  Red.pred <- Red.test.model_matrix[,names(Red.coeficientes_tam_iesimo)] %*% Red.coeficientes_tam_iesimo
  White.pred <- White.test.model_matrix[,names(White.coeficientes_tam_iesimo)] %*% White.coeficientes_tam_iesimo
  
  Red.MSE[i] <- mean((Red.test$quality - Red.pred)^2)
  White.MSE[i] <- mean((White.test$quality - White.pred)^2)
  
}

# Escoger el que ha obtenido el menor error cuadrático medio MSE.
print(Red.MSE)
print(White.MSE)






