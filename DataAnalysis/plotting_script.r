

test_data = read.csv("DataTable.csv")

# h size
h_mask = which(test_data$EvalName=="SizeH")
x_data = test_data$h_size[h_mask]
y_data = test_data$avg_times[h_mask]
plot(x_data, y_data, xlab="Hidden Layer Size", ylab="Average Lap Times", col="blue", pch=16, cex=2)
y_data = test_data$success_rate[h_mask]
plot(x_data, y_data, xlab="Hidden Layer Size", ylab="Average Lap Times", col="blue", pch=16, cex=2)

# beams
beam_mask = which(test_data$EvalName=="Beams")
plot(test_data$n_beams[beam_mask], test_data$success_rate[beam_mask], xlab="Number of Beams", ylab="Success Rate", col="red", pch=16, cex=2)
plot(test_data$n_beams[beam_mask], test_data$avg_times[beam_mask], xlab="Number of Beams", ylab="Average Times", col="red", pch=16, cex=2)
plot(test_data$n_beams[beam_mask], test_data$success_rate[beam_mask], xlab="Number of Beams", ylab="Success Rate", col="red", pch=16, cex=2, ylim=c(90, 100))
plot(test_data$n_beams[beam_mask], test_data$avg_times[beam_mask], xlab="Number of Beams", ylab="Average Times", col="red", pch=16, cex=2, ylim=c(350, 500))

# training steps
step_mask = which(test_data$EvalName=="TrainingSteps")
x_data = test_data$train_n[step_mask]
y_data = test_data$avg_times[step_mask]
plot(x_data, y_data, xlab="Training Steps", ylab="Average Lap Times", col="blue", pch=16, cex=2)

# zoomed steps
x_data = test_data$train_n[step_mask]
y_data = test_data$success_rate[step_mask]
plot(x_data, y_data, xlab="Training Steps", ylab="Success Rate", xlim=c(0,50000), col="red", pch=16, cex=2)
