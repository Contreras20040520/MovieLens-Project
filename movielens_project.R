# HarvardX: 9x Data Science Capstone
# MovieLens Project: Movie Rating Prediction
# Author: Juan Luis Rojas Contreras
# Date: July 2026

library(tidyverse)
library(caret)
library(data.table)
library(knitr)

options(timeout = 120)

RMSE <- function(true_values, predicted_values) sqrt(mean((true_values - predicted_values)^2))

dl <- "ml-10M100K.zip"
if(!file.exists(dl)) download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file)) unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file)) unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>% mutate(across(everything(), as.integer), rating = as.numeric(rating))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>% mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

final_holdout_test <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")
edx <- bind_rows(edx, anti_join(temp, final_holdout_test))
rm(dl, ratings, movies, test_index, temp, movielens)

save(edx, final_holdout_test, file = "movielens_data.RData")

set.seed(1, sample.kind = "Rounding")
validation_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-validation_index, ]
validation_set <- edx[validation_index, ] %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")
train_set <- bind_rows(train_set, anti_join(edx[validation_index, ], validation_set))
rm(validation_index)

print("Dataset dimensions")
print(paste("Training set:", nrow(train_set), "rows"))
print(paste("Validation set:", nrow(validation_set), "rows"))
print(paste("Final holdout test:", nrow(final_holdout_test), "rows"))
print(paste("Unique movies in training set:", length(unique(train_set$movieId))))
print(paste("Unique users in training set:", length(unique(train_set$userId))))

mu <- mean(train_set$rating)
pred_global <- rep(mu, nrow(validation_set))
rmse_global <- RMSE(validation_set$rating, pred_global)

rmse_results <- tibble(method = "Global mean model", RMSE = rmse_global)

movie_effects <- train_set %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu), .groups = "drop")
pred_movie <- validation_set %>% left_join(movie_effects, by = "movieId") %>% mutate(pred = mu + b_i) %>% pull(pred)
rmse_movie <- RMSE(validation_set$rating, pred_movie)

rmse_results <- bind_rows(rmse_results, tibble(method = "Movie effect model", RMSE = rmse_movie))

user_effects <- train_set %>% left_join(movie_effects, by = "movieId") %>% group_by(userId) %>% summarize(b_u = mean(rating - mu - b_i), .groups = "drop")
pred_movie_user <- validation_set %>% left_join(movie_effects, by = "movieId") %>% left_join(user_effects, by = "userId") %>% mutate(pred = mu + b_i + b_u) %>% pull(pred)
rmse_movie_user <- RMSE(validation_set$rating, pred_movie_user)

rmse_results <- bind_rows(rmse_results, tibble(method = "Movie + user effect model", RMSE = rmse_movie_user))

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(lambda) {
  mu_l <- mean(train_set$rating)
  b_i <- train_set %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu_l) / (n() + lambda), .groups = "drop")
  b_u <- train_set %>% left_join(b_i, by = "movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - mu_l - b_i) / (n() + lambda), .groups = "drop")
  validation_set %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% mutate(pred = mu_l + b_i + b_u) %>% pull(pred) %>% RMSE(validation_set$rating)
})

lambda_opt <- lambdas[which.min(rmses)]
rmse_regularized <- min(rmses)

rmse_results <- bind_rows(rmse_results, tibble(method = "Regularized movie + user effect model", RMSE = rmse_regularized))

print("Validation results summary")
print(knitr::kable(rmse_results, digits = 5))

mu_final <- mean(edx$rating)
b_i_final <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu_final) / (n() + lambda_opt), .groups = "drop")
b_u_final <- edx %>% left_join(b_i_final, by = "movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - mu_final - b_i) / (n() + lambda_opt), .groups = "drop")

pred_final <- final_holdout_test %>%
  left_join(b_i_final, by = "movieId") %>%
  left_join(b_u_final, by = "userId") %>%
  mutate(b_i = replace_na(b_i, 0), b_u = replace_na(b_u, 0), pred = mu_final + b_i + b_u) %>%
  pull(pred)

rmse_final <- RMSE(final_holdout_test$rating, pred_final)

final_result <- tibble(model = "Regularized movie + user effect model", lambda = lambda_opt, RMSE = rmse_final)

print("Final model results")
print(knitr::kable(final_result, digits = 5))

save(edx, final_holdout_test, train_set, validation_set, rmse_results, lambdas, rmses, lambda_opt, rmse_final, mu_final, b_i_final, b_u_final, final_result, file = "movielens_project_workspace.RData")

print(paste("Optimal lambda:", lambda_opt))
print(paste("Final RMSE on final_holdout_test:", round(rmse_final, 5)))
print("Target RMSE for full points: < 0.86490")

