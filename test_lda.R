dataset = read.csv('lda.csv')

dataset$Result = factor(dataset$Result, levels = c('P', 'F'), 
                        labels = c(1, 2))

X = dataset[c(1,2)]

N = NULL
N[1] = length(which(dataset$Result == 1))
N[2] = length(which(dataset$Result == 2))

X1 = subset(dataset, Result == 1)
X2 = subset(dataset, Result == 2)
X1 = X1[c(1:2)]
X2 = X2[c(1:2)]

mu1 = c(sum(X1$Curvature)/N[1], sum(X1$Diameter)/N[1])
mu2 = c(sum(X2$Curvature)/N[2], sum(X2$Diameter)/N[2])

mu = c(sum(dataset$Curvature)/7, sum(dataset$Diameter)/7)

X1[, 1] = X1[, 1] - mu[1]
X1[, 2] = X1[, 2] - mu[2]
X2[, 1] = X2[, 1] - mu[1]
X2[, 2] = X2[, 2] - mu[2]

C1 = as.matrix(t(X1)) %*% as.matrix(X1) / N[1]
C1 = as.data.frame(C1)
C2 = as.matrix(t(X2)) %*% as.matrix(X2) / N[2]
C2 = as.data.frame(C2)

C = data.frame(c(0,0), c(0,0))
for (i in 1:2)
{
  for (j in 1:2)
  {
    C[i,j] = (N[1] * C1[i,j] + N[2] * C2[i,j])/ 7
  }
}

C = inv(as.matrix(C))
C = as.data.frame(C)

p = NULL
p[1] = N[1]/nrow(dataset)
p[2] = N[2]/nrow(dataset)


#f1
a = as.matrix(t(mu1)) %*% as.matrix(C) %*% as.matrix(t(X[2,])) 
b = 0.5 * as.matrix(t(mu1)) %*% as.matrix(C) %*% as.matrix(mu1) + log(p[1])
f1 = a - b

a = as.matrix(t(mu2)) %*% as.matrix(C) %*% as.matrix(t(X[2,])) 
b = 0.5 * as.matrix(t(mu2)) %*% as.matrix(C) %*% as.matrix(mu2) + log(p[2])
f2 = a - b

#f2
a = as.matrix(t(mu2)) %*% as.matrix(C) %*% as.matrix(t(X)) 
b = 0.5 * as.matrix(t(mu2)) %*% as.matrix(C) %*% as.matrix(mu2) + log(p[2])
f2 = as.matrix(t(a)) - b[1]

f = data.frame(c(f1), c(f2))

#coeff
w1 = -as.matrix(C) %*% as.matrix(mu2)
w0 = -0.5 * as.matrix(t(mu2)) %*% as.matrix(C) %*% as.matrix(mu2) + log(p[2])

# Applying LDA
library(MASS)
dataset[1:2] = scale(dataset[1:2])
lda = lda(formula = Result ~ ., data = dataset)
dataset = as.data.frame(predict(lda, dataset))

