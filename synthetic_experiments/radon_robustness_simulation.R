set.seed(11905150)


#================= Helper Functions =================
# These functions do just simulate the computation

#' method simulating the breakdown behaviour of the singular radon point.
#' Assumes correct number of points for radon computation have been given.
radon_point_broken_down <- function(num_outliers) {

    return(sum(num_outliers)>=2)
}

#' simulates the behaviour of the whole radon machine up to the last level.
#' Assumes the correct number of points have been given.
#' base_hypotheses is a boolean vector, where a 1 indicates an outlier
#' returns the corresponding vector at the last layer
radon_machine <- function(height, base_hypotheses, radon_number=3, shuffle=T){
    res <- base_hypotheses
    for(i in rev(2:height)){
        res <- ifelse(shuffle,sample(res),res)
        # splits the hypotheses into radon-number large groups
        # seq_along(res) is like pythons range
        # with the division and ceil the first r numbers are 1 etc.
        splitx <- split(res,ceiling(seq_along(res)/radon_number))
        res <- sapply(splitx, radon_point_broken_down)
    }
    return(res)
}

#================Statistical Functions=========================

#' given n hypotheses in a level of the radon machine with num_outliers, calculates the probability
#' any random selection of radon_number many points contains at least 2 outliers (breaks down)
#' this function works only if the number of outliers is integer
prob_two_or_more_integer <- function(n, num_outliers, radon_number){
    1- sum(dhyper(0:1, num_outliers, n-num_outliers, radon_number))
}

# extendes the dhyper-function to fraktional populations, assuming that n is integer
dhyper_frak <- function(x,o,n,r){
    max(choose(o,x) * choose(n-o,r-x)/choose(n,r),0)
}



# this is most likely due to numerical precision: the sum sometimes is more than 1
prob_two_or_more_frak <- function(n,o,d){
    1- min(dhyper_frak(0,o,n,d) + dhyper_frak(1,o,n,d),1)
}




radon_exp<- function(d,h,o){
    if(h > 1)
        for(i in rev(2:h)){
            o <- prob_two_or_more_frak(d^i,o,d)*d^(i-1)
            # print(paste("r ",r," n ",d^i))
        }
    o
    #odds any final ball is a red one
}

radon_exp(5,3,2)

h <- 4
d <- 3
runs <- 200
x <- c()
y <- c()
for(o in 0:(d^h)){
    x[o+1] <- sum(replicate(runs,radon_machine(h, c(rep(1,o),rep(0,d^h-o)), radon_number = d)))/runs
    y[o+1] <- radon_exp(d,h,o)
    if(o>10)
       if(x[o-10]>d-0.1){
            break
       }
}
plot(x,y)
# plot(x-y)


o <- 5
res <- sample(c(rep(1,o),rep(0,d^h-o)))
splitx <- split(res,ceiling(seq_along(res)/(d)))
        # print(splitx)
res <- sapply(splitx, radon_point_broken_down)
res