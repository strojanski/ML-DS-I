
# Set working directory
getwd()
setwd("C:/Users/sebas/one/OneDrive/grive/faks/masters/y1/2nd semester/ML-DS I/ML-DS-I/hw2/")

# install.packages("ggplot2")
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
  
  temp_data <- df %>%
    select(logscore = all_of(logscore_col), distance = all_of(distance_col)) %>%
    mutate(Pair = paste("Fold", i))  # Add a column to identify the pair
  
  plot_data <- bind_rows(plot_data, temp_data)
}

# Plot all pairs on one graph
p <- ggplot(plot_data, aes(x = distance, y = logscore, color = Pair)) +
  geom_point(size = 2, alpha = 0.5) +  # Set point size and transparency
  labs(
    title = "Log score vs Distance for all folds", 
    x = "Distance", 
    y = "Log score",
    color = "Fold"  # Legend title
  ) +
  theme_minimal() +  # Minimal theme
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),  # Center and style the title
    axis.title = element_text(size = 12),  # Increase axis title size
    axis.text = element_text(size = 10),  # Increase axis tick label size
    legend.position = "bottom"  # Move legend to the bottom
  )

print(p)

# Part 2: Inverse encoding
comp_map <- c("EURO", "NBA", "SLO1", "U14", "U16")
 
# Unencode
for (pair in pairs) {
  col_name <- pair[3]  # Extract column name
  print(col_name)
  df[[col_name]] <- comp_map[df[[col_name]] + 1]
}

# For each split
for (pair in pairs) {
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
}
