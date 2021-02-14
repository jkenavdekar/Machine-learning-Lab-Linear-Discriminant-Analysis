library(datasets)

data(iris)
summary(iris)

library(ggplot2)
ggplot(iris, aes(x = Sepal.Width, y = Sepal.Length, col = Species)) + 
  geom_point()

iris$Species = factor(iris$Species, levels = c('setosa', 'versicolor', 'virginica'), 
                      labels = c(1, 2, 3))

dataset = iris[c(1,2,5)]

#Scale the features
#dataset[1:2] = scale(dataset[1:2])

#Manual Calculations for LDA

X = dataset[c(1,2)]

N = NULL
N[1] = length(which(dataset$Species == 1))
N[2] = length(which(dataset$Species == 2))
N[3] = length(which(dataset$Species == 3))

X1 = subset(dataset, Species == 1)
X2 = subset(dataset, Species == 2)
X3 = subset(dataset, Species == 3)
X1 = X1[c(1:2)]
X2 = X2[c(1:2)]
X3 = X3[c(1:2)]

mu1 = c(sum(X1$Sepal.Length)/N[1], sum(X1$Sepal.Width)/N[1])
mu2 = c(sum(X2$Sepal.Length)/N[2], sum(X2$Sepal.Width)/N[2])
mu3 = c(sum(X3$Sepal.Length)/N[3], sum(X3$Sepal.Width)/N[3])

C1 = cov(X1)
C2 = cov(X2)
C3 = cov(X3)

p = NULL
p[1] = N[1]/nrow(dataset)
p[2] = N[2]/nrow(dataset)
p[3] = N[3]/nrow(dataset)

K = length(unique(dataset$Species))

C = data.frame(c(0,0), c(0,0))
for (i in 1:(K-1))
  {
  for (j in 1:(K-1))
    {
      C[i,j] = (N[1] * C1[i,j] + N[2] * C2[i,j] + N[3] * C3[i,j]) / 150
    }
}

C
C = solve(C)

a = as.matrix(X) %*% as.matrix(C) %*% as.matrix(mu1)
b = 0.5 * as.matrix(t(mu1)) %*% as.matrix(C) %*% as.matrix(mu1)
f1 = a - as.vector(b) + log(p[1])

a = as.matrix(X) %*% as.matrix(C) %*% as.matrix(mu2)
b = 0.5 * as.matrix(t(mu2)) %*% as.matrix(C) %*% as.matrix(mu2)
f2 = a - as.vector(b) + log(p[2])

a = as.matrix(X) %*% as.matrix(C) %*% as.matrix(mu3)
b = 0.5 * as.matrix(t(mu3)) %*% as.matrix(C) %*% as.matrix(mu3)
f3 = a - as.vector(b) + log(p[3])

f <- NULL
f = data.frame(c(f1), c(f2), c(f3))
f$class <- apply(f,1,max)

for (i in 1:150)
{
  if(f[i,1] == f[i,4])
    f[i,4] = 1
  
  if(f[i,2] == f[i,4])
    f[i,4] = 2
  
  if(f[i,3] == f[i,4])
    f[i,4] = 3
}

plot(iris$Sepal.Width, iris$Sepal.Length,
     col=as.numeric(f$class)+1, pch=21,
     bg=as.numeric(f$class)+1,
     main="Calculated Prediction without lda()")

cm = table(f$class, dataset$Species)

r1 = sum(diag(table(f$class, dataset$Species)))
r2 = sum(table(f$class, dataset$Species))
(r2-r1)/r2


y1 = NULL
y2 = NULL
y3 = NULL
#computing equations of decision boundaries.
w01 = -0.5 * as.matrix(t(mu1)) %*% as.matrix(C) %*% as.matrix(mu1) + log(p[1])
w1 = as.matrix(C) %*% as.matrix(mu1)
y1 = as.matrix(X) %*% as.matrix(w1) + as.vector(w01)


w02 = -0.5 * as.matrix(t(mu2)) %*% as.matrix(C) %*% as.matrix(mu2) + log(p[2])
w2 = as.matrix(C) %*% as.matrix(mu2)
y2 = as.matrix(X) %*% as.matrix(w2) + as.vector(w02)


w03 = -0.5 * as.matrix(t(mu3)) %*% as.matrix(C) %*% as.matrix(mu3) + log(p[3])
w3 = as.matrix(C) %*% as.matrix(mu3)
y3 = as.matrix(X) %*% as.matrix(w3) + as.vector(w03)

f = data.frame(y1, y2, y3)
f$class <- apply(f, 1, which.max)

#Scale the features
dataset[1:2] = scale(dataset[1:2])

# Applying LDA
library(MASS)
lda = lda(formula = Species ~ ., data = dataset)
dataset = as.data.frame(predict(lda, dataset))
dataset = dataset[c(5, 6, 1)]

table(dataset$class, iris$Species)

#LDA prediction results
plot(iris$Sepal.Width, iris$Sepal.Length,
     col=as.numeric(dataset$class)+1, pch=21,
     bg=as.numeric(dataset$class)+1,
     main="Prediction using Lda()")

r1 = sum(diag(table(dataset$class, iris$Species)))
r2 = sum(table(dataset$class, iris$Species))
(r2-r1)/r2


#Q4
#sampled data
set.seed(1234)
library(mvtnorm)

sample(c(1, 2, 3), size = 150, replace= TRUE, c(0.333, 0.333, 0.333))

Q1 = as.data.frame(rmvnorm(50, mu1, C1))
Q1$Species = 1
Q2 = as.data.frame(rmvnorm(50, mu2, C2))
Q2$Species = 2
Q3 = as.data.frame(rmvnorm(50, mu3, C3))
Q3$Species = 3

Q <- rbind(Q1, Q2, Q3)
ggplot(Q, aes(x = V1, y = V2, col = as.factor(Species))) + geom_point()


#Q5
library(nnet)
dataset = iris[c(1,2,5)]
model <- multinom(Species ~ Sepal.Width + Sepal.Length, data=dataset)
summary(model)

Y = as.data.frame(model$fitted.values)
Y$class <- apply(Y,1,max)

for (i in 1:150)
{
  if(Y[i,1] == Y[i,4])
    Y[i,4] = 1
  
  if(Y[i,2] == Y[i,4])
    Y[i,4] = 2
  
  if(Y[i,3] == Y[i,4])
    Y[i,4] = 3
}

table(dataset$Species, Y$class)
r1 = sum(diag(table(dataset$Species, Y$class)))
r2 = sum(table(dataset$Species, Y$class))
(r2-r1)/r2

#Results Plot
plot(iris$Sepal.Width, iris$Sepal.Length,
     col=as.numeric(Y$class)+1, pch=21,
     bg=as.numeric(Y$class)+1,
     main="Logistic Regression")


###################################################################################


# My Test
#dataset = iris[c(1,2,5)]
library(e1071)
classifier = svm(formula = class ~ .,
                 data = dataset,
                 type = 'C-classification',
                 kernel = 'linear')



# Visualising the Training set results
library(ElemStatLearn)
set = dataset
g1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
g2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(g1, g2)
colnames(grid_set) = c('Sepal.Length', 'Sepal.Width')

#y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Decision Boundry', 
     xlab = 'Sepal.Length', ylab = 'Sepal.Width', 
     xlim = range(g1), ylim = range(g2))


y1 = as.data.frame(y1)


contour(g1, g2, matrix(as.numeric(y1), length(g1), length(g2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))
