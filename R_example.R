##install.packages(c("keras",""geoR))
library(keras)
library(geoR)
###
set.seed(123)
N = 1000
sim = grf(N, ny=1, cov.pars=c(1, 0.25), nug=0)
s = sim$coords[,1]
y = sim$data
image(sim, type="l",xlab='s',ylab='y')
num_basis = c(10,19,37,73)
##Wendland kernel
K = 0 ## basis size
phi = matrix(0,N, sum(num_basis))
for (res in 1:4){
    theta = 1/num_basis[res]*2.5
    len = num_basis[res]
    knots = (1:len)/len
    for (i in 1:len){
        d = abs(s-knots[i])/theta
        for (j in 1:length(d)){
            if (d[j] >= 0 && d[j] <= 1){
                phi[j,i + K] = (1-d[j])^6 * (35 * d[j]^2 + 18 * d[j] + 3)/3
            }
            else {
                phi[j,i + K] = 0
            }
        }
    }
    K = K + num_basis[res]
}



model <- keras_model_sequential() 
model %>% 
    layer_dense(units = 100, activation = 'relu', input_shape = c(K)) %>% 
    layer_dropout(rate = 0.2) %>% 
    layer_dense(units = 100, activation = 'relu') %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = 'linear')

model %>% compile(
    loss = 'mse',
    optimizer = optimizer_adam(),
    metrics = c('mse')
)

history <- model %>% fit(
    phi, y, 
    epochs = 200, batch_size = 128, 
    validation_split = 0.2
)

plot(history)

y_pred = model %>% predict(phi)
plot(s,y,type='p')
points(s,y_pred,pch=16)
