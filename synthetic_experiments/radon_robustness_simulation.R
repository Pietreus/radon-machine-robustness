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
        res <- base::ifelse(rep(shuffle,length(res)),sample(res, size= length(res)),res)
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

#' computes a generalized hypergeometric probability mass function,
#' where the num_outliers are permitted to be non-integer.
#' the function uses log_probabilities to improve numerical stability.
dhyper_frak_log <- function(wanted_successes, n, num_outliers, radon_number){
    log_choose <- function(a, b) {
        lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
    }
    log_prob <-
      log_choose(num_outliers, wanted_successes) +
        log_choose(n-num_outliers, radon_number - wanted_successes) -
        log_choose(n,radon_number)

    return(max(0,exp(log_prob)))
}


# this is most likely due to numerical precision: the sum sometimes is more than 1
prob_two_or_more_frak <- function(n,o,d){
    1- min(dhyper_frak(0,o,n,d) + dhyper_frak(1,o,n,d),1)
}

prob_two_or_more_frak_log <- function(n, num_outliers, radon_number){
    1- min(dhyper_frak_log(0, n, num_outliers, radon_number) + dhyper_frak_log(1, n, num_outliers, radon_number), 1)
}





radon_machine_outlier_expectation<- function(radon_number, height, num_outliers){
    if(height > 1)
        for(i in rev(2:height)){
            num_outliers <- prob_two_or_more_frak_log(radon_number^i, num_outliers, radon_number)*radon_number^(i-1)
        }
    num_outliers
}

# ===================== Sanity Checks =====================

# does the dhyper_frak function behave as probability mass function?
r <- 5
sum(sapply(0:r,function(x)dhyper_frak_log(x,r^5,2.7,r)))

# the old seems to as well, but now i can sleep better
# but it does not work well with large numbers!
sum(sapply(0:r,function(x)dhyper_frak(x,2.7,r^5,r)))

#careful: with non-integers + large numbers, they both become unstable/inaccurate
sum(sapply(0:25,function(x)dhyper_frak_log(x,25^10,2.7,25)))

# does the thing work equivalent to the normal one on integers?
sapply(0:r,function(x)dhyper_frak_log(x,r^5,5,r))
sapply(0:r,function(x)dhyper(x,5,r^5,r))
# can see inaccuracies but yes


# does the outlier expectation produce results similar to repeated simulation

radon_machine_outlier_expectation(5, 3, 20)

h <- 4
d <- 3
runs <- 200
x <- c()
y <- c()
for(o in 0:(d^h)){
    x[o+1] <- sum(replicate(runs,radon_machine(h, c(rep(1,o),rep(0,d^h-o)), radon_number = d)))/runs
    y[o+1] <- radon_machine_outlier_expectation(d, h, o)
    if(o>10)
       if(x[o-10]>d-0.1){
            break
       }
}
plot(x,y)
abline(0,1)
# plot(x-y)


o <- 5
res <- sample(c(rep(1,o),rep(0,d^h-o)))
splitx <- split(res,ceiling(seq_along(res)/(d)))
        # print(splitx)
res <- sapply(splitx, radon_point_broken_down)
res