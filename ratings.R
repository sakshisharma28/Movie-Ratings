library(dplyr)
library(ggplot2)
library(stringr)
library(tidyr)
library(stats)

ratings <- read.csv('ml-latest-small/ratings.csv')
movies <- read.csv('ml-latest-small/movies.csv')


moviesRatings <- movies %>% 
  left_join(ratings, by =  'movieId') %>% 
  mutate(genres = as.character(genres)) %>% 
  filter(!(is.na(userId))) %>% 
  group_by(userId, movieId, genres) %>% 
  summarise(rating = mean(rating)) %>% 
  ungroup()


avgMovieRating <- moviesRatings %>% 
  group_by(movieId) %>% 
  summarise(avgMRating = mean(rating)) %>% 
  ungroup()

avgUserRating <- moviesRatings %>% 
  group_by(userId) %>% 
  summarise(avgURating = mean(rating)) %>% 
  ungroup()

## Top 6 genres that people watched

Top6genres <- moviesRatings %>% select(userId, rating, genres) %>%
  mutate(gen = strsplit(moviesRatings$genres,"\\|"))  %>%
  unnest(gen) %>%
  group_by(gen) %>% 
  summarise(count = n()) %>% 
  ungroup() %>% 
  arrange(desc(count)) %>% 
  slice(1:6)
  
UserGenresRatings <- moviesRatings %>% select(userId, rating, genres) %>%
  mutate(gen = strsplit(moviesRatings$genres,"\\|"))  %>%
  unnest(gen) %>% 
  group_by(userId, gen) %>% 
  summarise(avgRating = mean(rating)) %>% 
  ungroup()

## rating distribution by genre

UserGenresRatings %>% 
  inner_join(Top6genres %>% select(gen), by = 'gen') %>% 
  ggplot(aes(avgRating), group = gen) + geom_density(aes(fill = gen)) +  facet_wrap(~gen, nrow = 2)


moviesRatings <- moviesRatings %>% 



## dataset of unique movies


## Top 500 rated movies
allmovies <- moviesRatings %>% 
  group_by(title) %>% 
  summarise(count = n()) %>% 
  ungroup() %>% 
  select(title)

top500 <- moviesRatings %>% 
  group_by(title) %>% 
  summarise(count = n()) %>% 
  ungroup() %>% 
  arrange(desc(count)) %>%
  slice(1:500) %>%
  select(title)


avgUserRating500 <- moviesRatings %>% inner_join(top500, by = 'title') %>% 
  group_by(userId) %>% 
  summarise(avgURating = mean(rating)) %>% 
  ungroup()
## Train-validation-test split

set.seed(123)

top500ratings <- moviesRatings %>% inner_join(top500, by = 'title')
skeleton <- expand.grid(userId = unique(top500ratings$userId), title = unique(top500ratings$title))

newtop500 <- skeleton %>% left_join(top500ratings, by = c('userId','title')) %>% 
  left_join(avgUserRating, by = 'userId') %>% 
  mutate(rating = ifelse(is.na(rating),avgURating, rating)) %>% select(-avgURating)

# shuffle <- sample(nrow(top500ratings))
# top500ratings <- top500ratings[shuffle,]

train <- newtop500
# validation <- moviesRatings[20418:40837,]
# test <- moviesRatings[40838:nrow(top1000ratings),]


############# Training the model #######################33
## User - movie sparse matrix

t_userMovieRating <- train %>% select(userId, title, rating) %>% 
  spread(title, rating)

toCluster<- t_userMovieRating[,-1]


## k-means clustering

nK <- 50
MSE <- matrix(NA,nrow = nK,ncol = 2)
set.seed(123)
for (i in seq(1:nK)){ 
  movieCluster<-kmeans(toCluster,i)
  userClusters <- cbind(userId = t_userMovieRating$userId, data.frame(cluster = movieCluster$cluster))
  MSE[i,1] <- i
  MSE[i,2] <- movieCluster$tot.withinss
}

MSE <- as.data.frame(MSE)
colnames(MSE) <- c("K","train_MSE")
MSE_all <- MSE %>% gather(type, MSE,-K)

MSE %>% gather(type, MSE,-K) %>% ggplot(aes(K,MSE), group = type) + geom_line(aes(color = type))


## Validation: 50% of the users labels retained, 50% hidden
## the prediction will be the mean rating for that movie by the people in the cluster who hace
## already rated it

k = 20

movieCluster<-kmeans(toCluster,k, nstart = 20)
userCluster <- data.frame(userId = t_userMovieRating[,1], cluster = movieCluster$cluster)


val500 <- moviesRatings %>% 
  group_by(title) %>% 
  summarise(count = n()) %>% 
  ungroup() %>% 
  arrange(desc(count)) %>%
  slice(501:1000) %>%
  select(title)

valUserRatings <- moviesRatings %>% inner_join(val500, by = 'title') %>% 
  inner_join(userCluster, by = 'userId')
  
## Divinding val to create 50-50 seen-unseen split
set.seed(123)
shuffle = sample(nrow(valUserRatings))
valUserRatings <- valUserRatings[shuffle,]
split = nrow(valUserRatings)/2

val_labelled <- valUserRatings[1:split,]
Moviepredlabelled <- val_labelled %>% group_by(cluster, title) %>% 
  summarise(avgRating = mean(rating)) %>% 
  ungroup()

val_unlabelled <- valUserRatings[(split + 1):nrow(valUserRatings),]

val_unlabelled_temp <- val_unlabelled %>% mutate(Predicting = NA) %>% 
  left_join(Moviepredlabelled, by = c('cluster','title')) %>% 
  left_join(avgUserRating500, by = 'userId') %>% 
  mutate(Predicting = ifelse(is.na(avgRating), avgURating, avgRating)) %>% 
  select(-c(avgURating, avgRating))

val_MSE = sum((val_unlabelled_temp$rating - val_unlabelled_temp$Predicting)^2)/nrow(val_unlabelled_temp)

val_MSE



#### Baseline:

baseline <- val_unlabelled %>%  left_join(avgUserRating500, by = 'userId') %>% 
  mutate(sqdiff = (rating - avgURating)^2)
MSE_baseline <- sum(baseline$sqdiff)/nrow(baseline)
MSE_baseline

#################################################################################################
#################################################################################################

shuffle = sample(nrow(moviesRatings))
moviesRatings <- moviesRatings[shuffle,]
split = nrow(moviesRatings)/2
train <- moviesRatings[1:split,]
validation <- moviesRatings[(split+1):nrow(moviesRatings),]

t_avgRatingGenre <- train %>% select(userId, rating, genres) %>%
  mutate(genre = strsplit(train$genres,"\\|"))  %>%
  unnest(genre) %>% 
  group_by(userId, genre) %>% 
  summarise(avgGenreRating = mean(rating)) %>% 
  ungroup()

t_avgUserRating <- train %>% 
  group_by(userId) %>% 
  summarise(avgUserRating = mean(rating)) %>% 
  ungroup()

t_avgMovieRating <- train %>% 
  group_by(movieId) %>% 
  summarise(avgMovieRating = mean(rating)) %>% 
  ungroup()

## Baseline 1 - average user rating

v_pred_1 <- validation %>% left_join(t_avgUserRating, by = 'userId') %>% 
  left_join(t_avgMovieRating, by = 'movieId') %>% 
  mutate(pred = ifelse(is.na(avgUserRating), avgMovieRating, avgUserRating),
         sqDiff = (pred - rating)^2)

v_pred_2 <- validation %>% left_join(t_avgUserRating, by = 'userId') %>% 
  left_join(t_avgMovieRating, by = 'movieId') %>% 
  mutate(pred = ifelse(is.na(avgMovieRating), avgUserRating, avgMovieRating),
         sqDiff = (pred - rating)^2)

MSE_baseline1 = sum(v_pred_1$sqDiff)/nrow(v_pred_1)
MSE_baseline2 = sum(v_pred_2$sqDiff)/nrow(v_pred_2)

MSE_baseline1
MSE_baseline2


### Cluster based on average rating on genres

toClusterdata <- t_avgRatingGenre %>% filter(genre != '(no genres listed)') %>% 
  spread(genre, avgGenreRating) 


toCluster <- toClusterdata[,-1]

toCluster[is.na(toCluster)] <- 0


nK <- 50
t_MSE <- matrix(NA,nrow = nK,ncol = 2)
set.seed(123)
for (i in seq(1:nK)){ 
  movieCluster<-kmeans(toCluster,i, nstart = 20)
  # userClusters <- cbind(userId = t_userMovieRating$userId, data.frame(cluster = movieCluster$cluster))
  t_MSE[i,1] <- i
  t_MSE[i,2] <- movieCluster$tot.withinss
}

t_MSE <- as.data.frame(t_MSE)
colnames(t_MSE) <- c("K","train_MSE")

t_MSE %>%  ggplot(aes(K,train_MSE)) + geom_line()

### Validating
k= 10

movieCluster<-kmeans(toCluster,k)

clusterCenters = as.data.frame(movieCluster$centers)

clusterCenters <- clusterCenters %>% mutate(cluster = seq(k)) %>% 
  gather(genres, avgClusterGenreRating, -cluster)

userCluster <- data.frame(userId = toClusterdata[,1], cluster = movieCluster$cluster)

t_clusterMovieRating <- train %>% left_join(userCluster, by = 'userId') %>% 
  group_by(cluster, movieId) %>% 
  summarise(avgClusterMovieRating = mean(rating)) %>% 
  ungroup()

val_pred_3 <- validation %>% left_join(userCluster, by = 'userId') %>% 
  left_join(t_clusterMovieRating, by = c('cluster', 'movieId')) %>% 
  left_join(t_avgMovieRating, by = 'movieId') %>% 
  left_join(t_avgUserRating, by = 'userId') %>% 
  mutate(pred = ifelse(is.na(avgClusterMovieRating), 
                       ifelse(is.na(avgUserRating), avgMovieRating, avgUserRating), 
                       avgClusterMovieRating),
         sqDiff = (rating - pred)^2)

val_pred_4 <- validation %>% left_join(userCluster, by = 'userId') %>% 
  left_join(t_clusterMovieRating, by = c('cluster', 'movieId')) %>% 
  left_join(t_avgMovieRating, by = 'movieId') %>% 
  left_join(t_avgUserRating, by = 'userId') %>% 
  mutate(pred = ifelse(is.na(avgClusterMovieRating), 
                       ifelse(is.na(avgMovieRating), avgUserRating, avgMovieRating), avgClusterMovieRating),
         sqDiff = (rating - pred)^2)

MSE_3 <- sum(val_pred_3$sqDiff)/nrow(val_pred_3)
MSE_3

MSE_4 <- sum(val_pred_4$sqDiff)/nrow(val_pred_4)
MSE_4

val_pred_5 <- validation %>% 
  mutate(genres = strsplit(validation$genres,"\\|"))  %>%
  unnest(genres) %>% 
  left_join(userCluster, by = 'userId') %>%
  left_join(clusterCenters, by = c('cluster','genres')) %>% 
  group_by(userId, cluster, movieId) %>% 
  summarise(rating)
