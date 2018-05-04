## SVM ANALYSIS

# load packages
library(e1071)
library(mlbench)

# --------------------------------------------
# functions to return specificity and sensitivity
getSensitivity <- function(table) {
  return(table[1,3]/(table[1,3] + table[2,3]))
}

getSpecificity <- function(table) {
  return(table[4,3]/(table[3,3] + table[4,3]))
}

# --------------------------------------------
# start with Iris dataset
# load data
iris <- iris
head(iris)

# pairwise plot iris data
cols <- character(nrow(iris))
cols[] <- "black"
cols[iris$Species == "setosa"] <- "slateblue"
cols[iris$Species == "versicolor"] <- "grey25"
cols[iris$Species == "virginica"] <- "tomato2"
pairs(iris[,1:4], col = cols)

# partition X matrix and y response
X <- iris[,1:4]
y <- as.factor(iris$Species) # as factor

# sample with seed and partition training and test sets
set.seed(1) # set seed
trn.idx <- sample(1:nrow(iris), round(nrow(iris)*0.8, 0))
tst.idx <- (-trn.idx)
train <- iris[c(trn.idx),]
test <- iris[c(-trn.idx),]

X.trn <- X[c(trn.idx),]
y.trn <- y[trn.idx]
X.tst <- X[c(tst.idx),]
y.tst <- y[tst.idx]

# build model with linear kernel
svm.linear.out <- svm(Species ~., data = train, kernel = "linear")
svm.linear.out
head(svm.linear.out$SV) # display support vectors used for classification

# evaluate performance 
pred.train <- predict(svm.linear.out, X.trn) # training response
mean(pred.train == y.trn) # prediction accuracy
table(pred.train, y.trn) # confusion matrix

pred.test <- predict(svm.linear.out, X.tst) # test response
mean(pred.test == y.tst) 
table(pred.test, y.tst)

# build model with polynomial kernel
svm.poly.out <- svm(Species ~., data = train, kernel = "polynomial")
svm.poly.out
head(svm.poly.out$SV) # display support vectors used for classification

# evaluate performance 
pred.train <- predict(svm.poly.out, X.trn) # training response
mean(pred.train == y.trn) # prediction accuracy
table(pred.train, y.trn) # confusion matrix

pred.test <- predict(svm.poly.out, X.tst) # test response
mean(pred.test == y.tst) 
table(pred.test, y.tst)

## tuning parameters: kernel type, classification type, C (regularization term), gamma

## larger values of C (regularization coefficient) are more precise, but may overfit on training data
## smaller values of C are less precise and have larger error rate

## larger gamma considers points closest to determine margin
## inversely, smaller gammas consider points further away from "center" to determine margin

# vectors for tuning
kernels <- c("linear", "polynomial", "radial", "sigmoid")
types <- c("C-classification", "nu-classification")
gam.vect <- c(0.01, 0.1, 0.25, 0.5, 1, 5, 10)
c.vect <- c(0.01, 0.1, 0.5, 1, 2, 5)

# get data frame of grid search results
svm.df <- as.character(data.frame())
for (i in 1:length(kernels)) {
  for (j in 1:length(types)) {
    # tune svm
    set.seed(1)
    svm.out <- tune.svm(Species ~., data = train, gamma = gam.vect, cost = c.vect, kernel = kernels[i], type = types[j])
    best.params <- svm.out$best.parameters # best parameters from grid search
    set.seed(1)
    svm.best.mod <- svm(Species ~., data = train, gamma = best.params$gamma, cost = best.params$cost, kernel = kernels[i], type = types[j])
    
    pred.train <- predict(svm.best.mod, X.trn) # training response
    trn.acc <- mean(pred.train == y.trn) 
    # table(pred.train, y.trn)
    pred.test <- predict(svm.best.mod, X.tst) # test response
    tst.acc <- mean(pred.test == y.tst)
    # table(pred.test, y.tst)
    
    temp.vect <- c(kernels[i], types[j], best.params$gamma, best.params$cost, round(trn.acc,3), round(tst.acc,3), sum(svm.best.mod$nSV))
    svm.df <- rbind(svm.df, as.character(temp.vect))
  }
}
colnames(svm.df) <- c("kernel", "type", "best.gamma", "best.c", "training.acc", "test.acc", "num.sp.vects") # modify column names

svm.df

# --------------------------------------------
# trying with breast cancer dataset
# load data
bc <- read.csv(file = "Desktop/MA 373 STAT MODELING/DATA/wisconsin breast cancer.csv", header = T)
head(bc)

# clean data
bc <- bc[,-1] # get rid of id column
bc[,1] <- as.factor(bc[,1])
X <- bc[,c(2:ncol(bc))]
y <- bc[,1]
colnames(bc)[1] <- "type"

# pairwise plot iris data
cols <- character(nrow(bc))
cols[] <- "slateblue"
cols[bc$type == "M"] <- "tomato2"
dev.new()
pairs(bc[,2:10], col = cols)
plot(bc$X0.2776, bc$X0.07871, col = cols) # overlapping data plot

# sample with seed and partition training and test sets
set.seed(1) # set seed
trn.idx <- sample(1:nrow(bc), round(nrow(bc)*0.8, 0))
tst.idx <- (-trn.idx)
train <- bc[c(trn.idx),]
test <- bc[c(-trn.idx),]

X.trn <- X[c(trn.idx),]
y.trn <- y[trn.idx]
X.tst <- X[c(tst.idx),]
y.tst <- y[tst.idx]

# get data frame of grid search results
svm.df <- as.character(data.frame())
for (i in 1:length(kernels)) {
  for (j in 1:length(types)) {
    # tune svm
    set.seed(1)
    svm.out <- tune.svm(type ~., data = train, gamma = gam.vect, cost = c.vect, kernel = kernels[i], type = types[j])
    best.params <- svm.out$best.parameters # best parameters from grid search
    set.seed(1)
    svm.best.mod <- svm(type ~., data = train, gamma = best.params$gamma, cost = best.params$cost, kernel = kernels[i], type = types[j])
    
    pred.train <- predict(svm.best.mod, X.trn) # training response
    trn.acc <- mean(pred.train == y.trn) 
    # table(pred.train, y.trn)
    pred.test <- predict(svm.best.mod, X.tst) # test response
    tst.acc <- mean(pred.test == y.tst)
    # table(pred.test, y.tst)
    
    temp.vect <- c(kernels[i], types[j], best.params$gamma, best.params$cost, round(trn.acc,3), round(tst.acc,3), sum(svm.best.mod$nSV))
    svm.df <- rbind(svm.df, as.character(temp.vect))
  }
}
colnames(svm.df) <- c("kernel", "type", "best.gamma", "best.c", "training.acc", "test.acc", "num.sp.vects") # modify column names

# naive bayes model
nb.out <- naiveBayes(type ~., data = train)
nb.out$apriori # apriori classes

# predict training response
pred.train <- predict(nb.out, X.trn)
trn.acc <- mean(pred.train == y.trn) 
table(pred.train, y.trn)

# predict test response
pred.test <- predict(nb.out, X.tst)
tst.acc <- mean(pred.test == y.tst) 
table(pred.test, y.tst)

# get sensitivity and specificity
tpr <- getSensitivity(as.data.frame(table(pred.test, y.tst)))
tnr <- getSpecificity(as.data.frame(table(pred.test, y.tst)))

# --------------------------------------------
# try glass dataset
data("Glass")
glass <- Glass

# plot data
cols <- character(nrow(glass))
cols[] <- "black"
cols[glass$Type == 2] <- palette()[2]
cols[glass$Type == 3] <- palette()[3]
cols[glass$Type == 5] <- palette()[5]
cols[glass$Type == 6] <- palette()[6]
cols[glass$Type == 7] <- palette()[7]
pairs(Glass[,1:9], col = cols)

# sample with seed and partition training and test sets
X <- glass[1:9]
y <- glass$Type

set.seed(1) # set seed
trn.idx <- sample(1:nrow(glass), round(nrow(glass)*0.8, 0))
tst.idx <- (-trn.idx)
train <- glass[c(trn.idx),]
test <- glass[c(-trn.idx),]

X.trn <- X[c(trn.idx),]
y.trn <- y[trn.idx]
X.tst <- X[c(tst.idx),]
y.tst <- y[tst.idx]

# vectors for tuning
kernels <- c("linear", "polynomial", "radial", "sigmoid")
types <- c("C-classification")
gam.vect <- c(0.01, 0.1, 0.25, 0.5, 1, 5, 10)
c.vect <- c(0.01, 0.1, 0.5, 1, 2, 5)

# get data frame of grid search results
svm.df <- as.character(data.frame())
for (i in 1:length(kernels)) {
  for (j in 1:length(types)) {
    # tune svm
    set.seed(1)
    svm.out <- tune.svm(Type ~., data = train, gamma = gam.vect, cost = c.vect, kernel = kernels[i], type = types[j])
    best.params <- svm.out$best.parameters # best parameters from grid search
    set.seed(1)
    svm.best.mod <- svm(Type ~., data = train, gamma = best.params$gamma, cost = best.params$cost, kernel = kernels[i], type = types[j])
    
    pred.train <- predict(svm.best.mod, X.trn) # training response
    trn.acc <- mean(pred.train == y.trn) 
    # table(pred.train, y.trn)
    pred.test <- predict(svm.best.mod, X.tst) # test response
    tst.acc <- mean(pred.test == y.tst)
    # table(pred.test, y.tst)
    
    temp.vect <- c(kernels[i], types[j], best.params$gamma, best.params$cost, round(trn.acc,3), round(tst.acc,3), sum(svm.best.mod$nSV))
    svm.df <- rbind(svm.df, as.character(temp.vect))
  }
}
colnames(svm.df) <- c("kernel", "type", "best.gamma", "best.c", "training.acc", "test.acc", "num.sp.vects") # modify column names


