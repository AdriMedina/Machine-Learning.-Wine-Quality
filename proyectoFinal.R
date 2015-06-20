
library(leaps)

# Cargamos las tablas de los ficheros.
WineQualityRed <- read.csv("C:/Users/Adri/Documents/GitHub/Machine-Learning.-Wine-Quality/winequality-red.csv", sep=";")
WineQualityWhite <- read.csv("C:/Users/Adri/Documents/GitHub/Machine-Learning.-Wine-Quality/winequality-white.csv", sep=";")


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


# Vamos a elegir las variables que más se implican para calcular la calidad del vino usando cross-validation
regfit.fwd=regsubsets(quality~., data=BalanceWineRed, nvmax=11, method="forward")
summary(regfit.fwd)






