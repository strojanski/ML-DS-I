
# Set working directory
getwd()
setwd("C:/Users/sebas/one/OneDrive/grive/faks/masters/y1/2nd semester/ML-DS I/ML-DS-I/hw2/")

# install.packages("ggplot2")
# install.packages("tidyverse")

# Load the required package
library(tidyr)
library(ggplot2)
library(dplyr)

# Read data and transform to dataframe
df <- read.csv("log_distance_errors.csv")
df <- as.data.frame(df)
df

# Get fold names
pairs <- list()
for (i in 0:4) {
  pairs[[i+1]] <- c(paste("log_score", i, sep="_"),
                    paste("distance", i, sep="_"),
                    paste("comp", i, sep="_"))
}

pairs

# Separate plots
for (pair in pairs) {
  logscore <- pair[1]
  distance <- pair[2]
  
  p <- ggplot(df, aes_string(x = distance, y=logscore)) +
    geom_point() +
    labs(title=paste("Scatter")) +
    theme_minimal()
  
  print(p)
}

# Combine all pairs into one data frame
plot_data <- data.frame()

for (i in seq_along(pairs)) {
  pair <- pairs[[i]]
  logscore_col <- pair[1]
  distance_col <- pair[2]
  comp_col <- pair[3]
  
  temp_data <- df %>%
    select(logscore = all_of(logscore_col), 
           distance = all_of(distance_col), 
           competition = all_of(comp_col)) %>%
    mutate(Pair = paste("Fold", i))  # Add a column to identify the pair
  
  plot_data <- bind_rows(plot_data, temp_data)
}

plot_data[10:20, ]


# Plot all pairs on one graph
p <- ggplot(plot_data, aes(x = distance, y = logscore, color = Pair)) +
  geom_point(size = 2, alpha = 0.5) +  # Set point size and transparency
  labs(
    title = "", 
    x = "Distance", 
    y = "Log score",
    color = "Fold"
  ) +
  theme_minimal() +  
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),  
    axis.title = element_text(size = 12),  
    axis.text = element_text(size = 10),  
    legend.position = "bottom"
  )



p <- ggplot(plot_data, aes(x = distance, y = logscore, color = Pair)) +
  geom_point(size = 2, alpha = 0.5) + 
  geom_smooth(
    aes(x = distance, y = logscore, group = 1),  
    method = "loess",  # Use LOESS smoothing
    se = FALSE, 
    color = "black",  
    size = 2, 
    linetype = "solid" 
  ) +
  labs(
    title = "", 
    x = "Distance", 
    y = "Log score",
    color = "Fold"  
  ) +
  theme_minimal() + 
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),  
    axis.title = element_text(size = 12),  
    axis.text = element_text(size = 10),  
    legend.position = "bottom"  
  )

print(p)

# Bootstrap distance correlation:
set.seed(42) 
n_boot <- 1000  
boot_corr <- numeric(n_boot)

for (i in 1:n_boot) {
  sample_indices <- sample(nrow(plot_data), replace = TRUE)
  boot_sample <- plot_data[sample_indices, ]
  boot_corr[i] <- cor(boot_sample$distance, boot_sample$logscore, use = "complete.obs")
}

# Compute 95% confidence interval
quantile(boot_corr, c(0.025, 0.975))
sd(boot_corr)
mean(boot_corr)

# Part 2: Inverse encoding
comp_map <- c("EURO", "NBA", "SLO1", "U14", "U16")
 
# Unencode
for (pair in pairs) {
  col_name <- pair[3]  # Extract column name
  print(col_name)
  df[[col_name]] <- comp_map[df[[col_name]] + 1]
}

# For each split
weighted <- numeric(5)
unweighted <- numeric(5)
i <- 0
for (pair in pairs) {
  i <- i+1
  comp_column <- pair[3]  
  log_column <- pair[1] 
  print(comp_column)
  
  # Get empirical relative frequencies
  relative_freq <- table(df[[comp_column]]) / sum(!is.na(df[[comp_column]]))
  
  # Get mean log score for each type of competition
  logscore_means <- df %>%
    group_by(across(all_of(comp_column))) %>% 
    summarize(mean_logscore = mean(.data[[log_column]], na.rm = TRUE)) 
  
  # Drop NA rows
  logscore_means <- na.omit(logscore_means)
  logscore_means
  
  # Assign column names
  names(logscore_means) <- c(comp_column, "mean_logscore")
  
  # Define true DGF frequencies
  true_freqs <- c(0.1, 0.6, 0.1, 0.1, 0.1) 
  names(true_freqs) <- comp_map
  
  # Compute weighted log score
  weighted_log_score <- sum(logscore_means$mean_logscore * true_freqs[as.character(logscore_means[[comp_column]])])
  print("Weighted log score")
  print(weighted_log_score)
  
  # Compute original log score
  original_log_score <- mean(df[[log_column]], na.rm = TRUE)
  print("Empirical log score")
  print(original_log_score)
  cat("\n")
  weighted[i] <- weighted_log_score
  unweighted[i] <- original_log_score
}

mean(weighted)
sd(weighted)

mean(unweighted)
sd(unweighted)

# Get confidence intervals for the Competition
true_freqs <- c("EURO" = 0.1, "NBA" = 0.6, "SLO1" = 0.1, "U14" = 0.1, "U16" = 0.1)

df_all <- plot_data

boot_log_scores <- numeric(n_boot)
for (i in 1:n_boot) {
  boot_sample <- df_all[sample(nrow(df_all), replace = TRUE), ]
  
  # Compute mean log score for each competition type
  logscore_means <- boot_sample %>%
    group_by(competition) %>%
    summarize(mean_logscore = mean(logscore, na.rm = TRUE))
  
  # Merge with true frequencies
  logscore_means <- logscore_means %>%
    mutate(weight = true_freqs[as.character(competition)]) %>%
    drop_na(weight)  # Drop any missing weights
  
  boot_log_scores[i] <- sum(logscore_means$mean_logscore * logscore_means$weight, na.rm = TRUE)
}

boot_unweighted <- numeric(n_boot)

for (i in 1:n_boot) {
  # Resample with replacement
  boot_sample <- df_all[sample(nrow(df_all), replace = TRUE), ]
  
  # Compute unweighted log loss (simple mean)
  boot_unweighted[i] <- mean(boot_sample$logscore, na.rm = TRUE)
}

ci_unweighted <- quantile(boot_unweighted, c(0.025, 0.975), na.rm=FALSE)
ci_weighted <- quantile(boot_log_scores, c(0.025, 0.975))


cat("95% CI for Unweighted Log Score:", ci_unweighted, "\n")
cat("95% CI for Weighted Log Score:", ci_weighted, "\n")

