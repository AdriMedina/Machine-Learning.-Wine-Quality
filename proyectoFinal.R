
# Cargamos las tablas de los ficheros.
WineQualityRed <- read.csv("C:/Users/Adri/Desktop/AA/practicas/proyecto/winequality-red.csv", sep=";")
WineQualityWhite <- read.csv("C:/Users/Adri/Desktop/AA/practicas/proyecto/winequality-white.csv", sep=";")


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


# Vamos a elegir las variables que más se implican para calcular la calidad del vino usando cross-validation







