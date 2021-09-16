library(reshape2)
library(gridExtra)
library(ggplot2)

total_set = melt(data.frame(t(read.csv("RepeatingTimes_4.csv", header=F))))

layout(mat=matrix(c(1, 2), 2, 1, byrow=T))

par(mar=c(1, 1, 1, 1))
hist(total_set$value, xlim=c(360, 650), breaks=50)
par(mar=c(1, 1, 0, 1))
boxplot(total_set$value, ylim=c(370,700), horizontal = T, axes=F)


