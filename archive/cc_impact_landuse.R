# predict the change in conversion probability resulting from a changed climate
# restrict study area to eastern U.S.

library(tidyverse)
library(gridExtra)

calc_utility <- function(model, df) {
  
  df$crop_util <- model$coefficients[7] * df$nr.cr
  
  df$pasture_util <- model$coefficients[3] + model$coefficients[4] *
    df$lcc + model$coefficients[9] *
    df$nr.ps
  
  df$forest_util <- model$coefficients[1] + model$coefficients[2] *
    df$lcc + model$coefficients[8] *
    df$nr.fr
  
  df$urban_util <- model$coefficients[5] + model$coefficients[6] *
    df$lcc + model$coefficients[10] *
    df$nr.ur
  
  df
}

calc_prob_chg <- function(model, df1, df2) {
  
  df1 <- df1 %>%
    mutate(nr.cr = crnr_obs,
           nr.fr = frnr_obs,
           nr.ur = urnr_obs)
  
  df1 <- calc_utility(model, df1)
  df2 <- calc_utility(model, df2)
  
  df1 <- df1 %>%
    mutate(pr_crop0 = exp(crop_util) / (exp(crop_util) + exp(pasture_util) + 
                                         exp(forest_util) + exp(urban_util)),
           pr_pasture0 = exp(pasture_util) / (exp(crop_util) + exp(pasture_util) + 
                                               exp(forest_util) + exp(urban_util)),
           pr_forest0 = exp(forest_util) / (exp(crop_util) + exp(pasture_util) + 
                                             exp(forest_util) + exp(urban_util)),
           pr_urban0 = exp(urban_util) / (exp(crop_util) + exp(pasture_util) + 
                                           exp(forest_util) + exp(urban_util))) %>%
    select(fips, riad_id, xfact, starts_with('pr'))
  
  df2 <- df2 %>%
    mutate(pr_crop1 = exp(crop_util) / (exp(crop_util) + exp(pasture_util) + 
                                         exp(forest_util) + exp(urban_util)),
           pr_pasture1 = exp(pasture_util) / (exp(crop_util) + exp(pasture_util) + 
                                               exp(forest_util) + exp(urban_util)),
           pr_forest1 = exp(forest_util) / (exp(crop_util) + exp(pasture_util) + 
                                             exp(forest_util) + exp(urban_util)),
           pr_urban1 = exp(urban_util) / (exp(crop_util) + exp(pasture_util) + 
                                           exp(forest_util) + exp(urban_util))) %>%
    select(starts_with('pr'))
  
  df <- cbind(df1, df2) %>%
    mutate(probchg_crop = pr_crop1 - pr_crop0, 
           probchg_pasture = pr_pasture1 - pr_pasture0,
           probchg_forest = pr_forest1 - pr_forest0,
           probchg_urban = pr_urban1 - pr_urban0)
  
  df
}

aggr_to_county <- function(mfx_df) {
  df <- mfx_df %>%
    group_by(fips) %>%
    summarize(mfx_crop = weighted.mean(mfx_crop, w = xfact), 
              mfx_pasture = weighted.mean(mfx_pasture, w = xfact), 
              mfx_forest = weighted.mean(mfx_forest, w = xfact),
              mfx_urban = weighted.mean(mfx_urban, w = xfact)) %>%
    ungroup()
}

# load model parameters from land use model

# landuse_models <- readRDS('crop_forest_levels_2010_2012.rds')
map(landuse_models, ~ summary(.))
model_crstart <- landuse_models[[3]]
model_frstart <- landuse_models[[11]]
model_prstart <- landuse_models[[7]]

stargazer(model_crstart, model_frstart)

# load estimation data from logit model estimation

est_dat <- readRDS('estimation_data_crop_forest_logit_2010_2012.rds')

dat_crstart <- est_dat[['cropstart_east']]
dat_frstart <- est_dat[['foreststart_east']]

# format cc impact data---------------------------------------------------------

cc_crop <- tbl_df(readRDS('cc_impacts/crop_climate_change_impact.rds')) %>%
  mutate(fips = as.integer(GEOID)) %>%
  select(fips, nr, impact)
  
cc_forest <- tbl_df(readRDS('cc_impacts/forest_climate_change_impact.rds')) %>%
  mutate(fips = as.integer(GEOID)) %>%
  select(fips, nracre, impact)

cc_impact <- tbl_df(readRDS('cc_impacts/urban_climate_change_impact.rds')) %>%
  select(fips, urnr, impact) %>%
  full_join(cc_forest, by = 'fips') %>%
  full_join(cc_crop, by = 'fips') %>%
  rename('urnr_obs' = 'urnr', 'urnr_impact' = 'impact.x',
         'frnr_obs' = 'nracre', 'frnr_impact' = 'impact.y',
         'crnr_obs' = 'nr', 'crnr_impact' = 'impact')

# map cc impacts----------------------------------------------------------------

library(sf)
library(tmap)

counties <- st_read('shapefiles/conus_county.shp')

cc_mapd <- cc_impact
cc_mapd$fips <- factor(str_pad(cc_mapd$fips, 5, side = 'left', pad = '0'))

cc_mapd <- counties %>%
  right_join(cc_mapd, by = c('GEOID' = 'fips'))

map <- tm_shape(cc_mapd) +
  tm_fill('crnr_obs', palette = "RdYlGn", style = 'quantile',
          n = 5, title = 'Net Return ($2010)', legend.hist = T) +
  tm_layout(scale = 0.8, legend.position = c('right','bottom'),
            legend.outside = T) +
  tm_shape(cc_mapd) +
  tm_borders(lwd = .5, alpha = .5)

map <- tm_shape(cc_mapd) +
  tm_fill('frnr_impact', palette = "RdYlGn", style = 'quantile',
          n = 5, title = 'Net Return Impact ($2010)', legend.hist = T) +
  tm_layout(scale = 0.8, legend.position = c('right','bottom'),
            legend.outside = T) +
  tm_shape(cc_mapd) +
  tm_borders(lwd = .5, alpha = .5)

# prepare data for impact calculations------------------------------------------

startcr <- dat_crstart %>%
  inner_join(cc_impact, by = 'fips') %>%
  filter(region %in% c('SO'), year == 2012) %>%
  select(-starts_with('nrmean'), -starts_with('nrchange'),
         -subregion, -region, -weight, -year)

startfr <- dat_frstart %>%
  inner_join(cc_impact, by = 'fips') %>%
  filter(region %in% c('SO'), year == 2012) %>%
  select(-starts_with('nrmean'), -starts_with('nrchange'),
         -subregion, -region, -weight, -year)

#-------------------------------------------------------------------------------

# full impact on land starting in crop------------------------------------------
startcr_fullim <- startcr %>%
  mutate(nr.cr = crnr_obs + crnr_impact,
         nr.fr = frnr_obs + frnr_impact,
         nr.ur = urnr_obs + urnr_impact)

startcr_fullim <- calc_prob_chg(model_crstart, startcr, startcr_fullim) %>%
  select(fips, riad_id, xfact, starts_with('probchg')) %>%
  gather('landuse', 'probchg', 4:7) %>%
  mutate(impact_type = 'full_impact_startcr')

# full impact on land starting in forest----------------------------------------

startfr_fullim <- startfr %>%
  mutate(nr.cr = crnr_obs + crnr_impact,
         nr.fr = frnr_obs + frnr_impact,
         nr.ur = urnr_obs + urnr_impact)

startfr_fullim <- calc_prob_chg(model_frstart, startfr, startfr_fullim) %>%
  select(fips, riad_id, xfact, starts_with('probchg')) %>%
  gather('landuse', 'probchg', 4:7) %>%
  mutate(impact_type = 'full_impact_startfr')

# Crop impact on land starting in crop------------------------------------------

startcr_crim <- startcr %>%
  mutate(nr.cr = crnr_obs + crnr_impact)

startcr_crim <- calc_prob_chg(model_crstart, startcr, startcr_crim) %>%
  select(fips, riad_id, xfact, starts_with('probchg')) %>%
  gather('landuse', 'probchg', 4:7) %>%
  mutate(impact_type = 'crop_impact_startcr')

# crop impact on land starting in forest

startfr_crim <- startfr %>%
  mutate(nr.cr = crnr_obs + crnr_impact)

startfr_crim <- calc_prob_chg(model_frstart, startfr, startfr_crim) %>%
  select(fips, riad_id, xfact, starts_with('probchg')) %>%
  gather('landuse', 'probchg', 4:7) %>%
  mutate(impact_type = 'crop_impact_startfr')

# forest impact on land starting in crop------------------------------------------

startcr_frim <- startcr %>%
  mutate(nr.fr = frnr_obs + frnr_impact)

startcr_frim <- calc_prob_chg(model_crstart, startcr, startcr_frim) %>%
  select(fips, riad_id, xfact, starts_with('probchg')) %>%
  gather('landuse', 'probchg', 4:7) %>%
  mutate(impact_type = 'forest_impact_startcr')

# forest impact on land starting in forest

startfr_frim <- startfr %>%
  mutate(nr.fr = frnr_obs + frnr_impact)

startfr_frim <- calc_prob_chg(model_frstart, startfr, startfr_frim) %>%
  select(fips, riad_id, xfact, starts_with('probchg')) %>%
  gather('landuse', 'probchg', 4:7) %>%
  mutate(impact_type = 'forest_impact_startfr')

# urban impact on land starting in crop------------------------------------------

startcr_urim <- startcr %>%
  mutate(nr.ur = urnr_obs + urnr_impact)

startcr_urim <- calc_prob_chg(model_crstart, startcr, startcr_urim) %>%
  select(fips, riad_id, xfact, starts_with('probchg')) %>%
  gather('landuse', 'probchg', 4:7) %>%
  mutate(impact_type = 'urban_impact_startcr')

# urban impact on land starting in forest

startfr_urim <- startfr %>%
  mutate(nr.ur = urnr_obs + urnr_impact)

startfr_urim <- calc_prob_chg(model_frstart, startfr, startfr_urim) %>%
  select(fips, riad_id, xfact, starts_with('probchg')) %>%
  gather('landuse', 'probchg', 4:7) %>%
  mutate(impact_type = 'urban_impact_startfr')



# bind impact data--------------------------------------------------------------

d1 <- rbind(startcr_fullim, startfr_fullim) %>%
  mutate(type_landuse = paste(impact_type, landuse, sep = '_'))

d2 <- rbind(startcr_crim, startfr_crim) %>%
  mutate(type_landuse = paste(impact_type, landuse, sep = '_'))

d3 <- rbind(startcr_frim, startfr_frim) %>%
  mutate(type_landuse = paste(impact_type, landuse, sep = '_'))

d4 <- rbind(startcr_urim, startfr_urim) %>%
  mutate(type_landuse = paste(impact_type, landuse, sep = '_'))

d5 <- rbind(d1, d2, d3, d4) %>%
  filter(!is.na(probchg))

# create boxplot of climate impacts---------------------------------------------
boxd <- filter(d5, !is.na(probchg), probchg > -0.003, probchg < 0.003)

box1 <- ggplot(data = d5[d5$type_landuse %in% c('full_impact_startcr_probchg_crop',
                                        'crop_impact_startcr_probchg_crop',
                                        'forest_impact_startcr_probchg_crop',
                                        'urban_impact_startcr_probchg_crop') &
                   d5$probchg < 0.0003 &
                   d5$probchg > -0.003,]) +
  geom_boxplot(aes(type_landuse, probchg)) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(y = 'Change in Probability of Conversion',
       x = 'Landuse',
       title = 'Climate Change Impact on Conversion Probability') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank()) +
  coord_flip()

box2 <- ggplot(data = d5[d5$type_landuse %in% c('full_impact_startcr_probchg_forest',
                                        'crop_impact_startcr_probchg_forest',
                                        'forest_impact_startcr_probchg_forest',
                                        'urban_impact_startcr_probchg_forest') &
                           d5$probchg < 0.0005,]) +
  geom_boxplot(aes(type_landuse, probchg)) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(y = 'Change in Probability of Conversion',
       x = 'Landuse',
       title = 'Climate Change Impact on Conversion Probability') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank()) +
  coord_flip()

box3 <- ggplot(data = d5[d5$type_landuse %in% c('full_impact_startcr_probchg_pasture',
                                                'crop_impact_startcr_probchg_pasture',
                                                'forest_impact_startcr_probchg_pasture',
                                                'urban_impact_startcr_probchg_pasture') &
                           d5$probchg < 0.002 &
                           d5$probchg > -0.002,]) +
  geom_boxplot(aes(type_landuse, probchg)) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(y = 'Change in Probability of Conversion',
       x = 'Landuse',
       title = 'Climate Change Impact on Conversion Probability') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank()) +
  coord_flip()

box4 <- ggplot(data = d5[d5$type_landuse %in% c('full_impact_startcr_probchg_urban',
                                                'crop_impact_startcr_probchg_urban',
                                                'forest_impact_startcr_probchg_urban',
                                                'urban_impact_startcr_probchg_urban') &
                           d5$probchg < 0.005 &
                           d5$probchg > -0.005,]) +
  geom_boxplot(aes(type_landuse, probchg)) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(y = 'Change in Probability of Conversion',
       x = 'Landuse',
       title = 'Climate Change Impact on Conversion Probability') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank()) +
  coord_flip()

box5 <- ggplot(data = d5[d5$type_landuse %in% c('full_impact_startfr_probchg_crop',
                                                'crop_impact_startfr_probchg_crop',
                                                'forest_impact_startfr_probchg_crop',
                                                'urban_impact_startfr_probchg_crop') &
                           d5$probchg < 0.0001 &
                           d5$probchg > -0.0001,]) +
  geom_boxplot(aes(type_landuse, probchg)) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(y = 'Change in Probability of Conversion',
       x = 'Landuse',
       title = 'Climate Change Impact on Conversion Probability') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank()) +
  coord_flip()

box6 <- ggplot(data = d5[d5$type_landuse %in% c('full_impact_startfr_probchg_forest',
                                                'crop_impact_startfr_probchg_forest',
                                                'forest_impact_startfr_probchg_forest',
                                                'urban_impact_startfr_probchg_forest') &
                           d5$probchg < 0.005 &
                           d5$probchg > -0.005,]) +
  geom_boxplot(aes(type_landuse, probchg)) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(y = 'Change in Probability of Conversion',
       x = 'Landuse',
       title = 'Climate Change Impact on Conversion Probability') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank()) +
  coord_flip()

box7 <- ggplot(data = d5[d5$type_landuse %in% c('full_impact_startfr_probchg_pasture',
                                                'crop_impact_startfr_probchg_pasture',
                                                'forest_impact_startfr_probchg_pasture',
                                                'urban_impact_startfr_probchg_pasture') &
                           d5$probchg < 0.00005 &
                           d5$probchg > -0.0001,]) +
  geom_boxplot(aes(type_landuse, probchg)) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(y = 'Change in Probability of Conversion',
       x = 'Landuse',
       title = 'Climate Change Impact on Conversion Probability') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank()) +
  coord_flip()

box8 <- ggplot(data = d5[d5$type_landuse %in% c('full_impact_startfr_probchg_urban',
                                                'crop_impact_startfr_probchg_urban',
                                                'forest_impact_startfr_probchg_urban',
                                                'urban_impact_startfr_probchg_urban') &
                           d5$probchg < 0.003 &
                           d5$probchg > -0.003,]) +
  geom_boxplot(aes(type_landuse, probchg)) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(y = 'Change in Probability of Conversion',
       x = 'Landuse',
       title = 'Climate Change Impact on Conversion Probability') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank()) +
  coord_flip()

pdf('cc_impact_crstart_boxplot.pdf', width = 14)
grid.arrange(box1, box2, box3, box4)
dev.off()

pdf('cc_impact_frstart_boxplot.pdf', width = 14)
grid.arrange(box5, box6, box7, box8)
dev.off()

pdf('cc_impact_boxplot.pdf')
ggplot(data = boxd) +
  geom_boxplot(aes(type_landuse, probchg)) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(y = 'Change in Probability of Conversion',
       x = 'Landuse',
       title = 'Climate Change Impact on Conversion Probability') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank()) +
  coord_flip()
dev.off()
  

# histograms for climate impacts------------------------------------------------

h1 <- ggplot(startcr_fullim[startcr_fullim$landuse == 'probchg_crop',]) +
  geom_histogram(aes(x = probchg),
                 binwidth = .0005, color = 'black', fill = 'blue', alpha = .2) +
  ggtitle('Land Starting in Crop Use') +
  labs(x = 'Change in Probability\n of Remaining in Crop Use', y = 'Frequency') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

h2 <- ggplot(startcr_fullim[startcr_fullim$landuse == 'probchg_forest',]) +
  geom_histogram(aes(x = probchg),
                 binwidth = .00001, color = 'black', fill = 'blue', alpha = .2) +
  labs(x = 'Change in Probability\n of Converting to Forest Use', y = 'Frequency') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

h3 <- ggplot(startcr_fullim[startcr_fullim$landuse == 'probchg_urban',]) +
  geom_histogram(aes(x = probchg),
                 binwidth = .0005, color = 'black', fill = 'blue', alpha = .2) +
  labs(x = 'Change in Probability\n of Converting to Urban Use', y = 'Frequency') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

h4 <- ggplot(startcr_fullim[startcr_fullim$landuse == 'probchg_pasture',]) +
  geom_histogram(aes(x = probchg),
                 binwidth = .00025, color = 'black', fill = 'blue', alpha = .2) +
  labs(x = 'Change in Probability\n of Converting to Pasture Use', y = 'Frequency') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

h5 <- ggplot(startfr_fullim[startfr_fullim$landuse == 'probchg_crop' & 
                        startfr_fullim$probchg < 0.0005,]) +
  geom_histogram(aes(x = probchg),
                 binwidth = .00004, color = 'black', fill = 'blue', alpha = .2) +
  ggtitle('Land Starting in Forest Use') +
  labs(x = 'Change in Probability\n of Converting to Crop Use', y = 'Frequency') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

h6 <- ggplot(startfr_fullim[startfr_fullim$landuse == 'probchg_forest',]) +
  geom_histogram(aes(x = probchg),
                 binwidth = .0005, color = 'black', fill = 'blue', alpha = .2) +
  labs(x = 'Change in Probability\n of Remaining in Forest Use', y = 'Frequency') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

h7 <- ggplot(startfr_fullim[startfr_fullim$landuse == 'probchg_urban',]) +
  geom_histogram(aes(x = probchg),
                 binwidth = .0005, color = 'black', fill = 'blue', alpha = .2) +
  labs(x = 'Change in Probability\n of Converting to Urban Use', y = 'Frequency') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

h8 <- ggplot(startfr_fullim[startfr_fullim$landuse == 'probchg_pasture',]) +
  geom_histogram(aes(x = probchg),
                 binwidth = .00000085, color = 'black', fill = 'blue', alpha = .2) +
  labs(x = 'Change in Probability\n of Converting to Pasture Use', y = 'Frequency') +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

pdf('cc_impact_histograms.pdf')
grid.arrange(h1, h5, h2, h6,
             h3, h7, h4, h8,
             nrow = 4, ncol = 2)
dev.off()

# map impact results------------------------------------------------------------

mapd <- d5 %>%
  group_by(fips, landuse, impact_type, type_landuse) %>%
  summarize(probchg = weighted.mean(probchg, w = xfact)) %>%
  ungroup() %>%
  mutate(fips = factor(str_pad(fips, 5, side = 'left', pad = '0')))

mapd <- counties %>%
  right_join(mapd, by = c('GEOID' = 'fips'))

map1 <- tm_shape(mapd[mapd$type_landuse == 'full_impact_startcr_probchg_crop',]) +
  tm_fill('probchg', palette = 'RdYlGn', style = 'quantile',
          n = 5, title = 'Change in Probability\nof Remaining in Crop Use', legend.hist = F) +
  tm_layout(scale = 0.8, legend.position = c('right','bottom'),
            legend.outside = T) +
  tm_shape(mapd[mapd$type_landuse == 'full_impact_startcr_probchg_crop',]) +
  tm_borders(lwd = .5, alpha = .5)

map2 <- tm_shape(mapd[mapd$type_landuse == 'full_impact_startcr_probchg_forest',]) +
  tm_fill('probchg', palette = 'RdYlGn', style = 'quantile',
          n = 5, title = 'Change in Probability\nof Converting to Forest Use', legend.hist = F) +
  tm_layout(scale = 0.8, legend.position = c('right','bottom'),
            legend.outside = T) +
  tm_shape(mapd[mapd$type_landuse == 'full_impact_startcr_probchg_forest',]) +
  tm_borders(lwd = .5, alpha = .5)

map3 <- tm_shape(mapd[mapd$type_landuse == 'full_impact_startcr_probchg_urban',]) +
  tm_fill('probchg', palette = 'RdYlGn', style = 'quantile',
          n = 5, title = 'Change in Probability\nof Converting to Urban Use', legend.hist = F) +
  tm_layout(scale = 0.8, legend.position = c('right','bottom'),
            legend.outside = T) +
  tm_shape(mapd[mapd$type_landuse == 'full_impact_startcr_probchg_urban',]) +
  tm_borders(lwd = .5, alpha = .5)

map4 <- tm_shape(mapd[mapd$type_landuse == 'full_impact_startcr_probchg_pasture',]) +
  tm_fill('probchg', palette = 'RdYlGn', style = 'quantile',
          n = 5, title = 'Change in Probability\nof Converting to Pasture Use', legend.hist = F) +
  tm_layout(scale = 0.8, legend.position = c('right','bottom'),
            legend.outside = T) +
  tm_shape(mapd[mapd$type_landuse == 'full_impact_startcr_probchg_pasture',]) +
  tm_borders(lwd = .5, alpha = .5)



map5 <- tm_shape(mapd[mapd$type_landuse == 'full_impact_startfr_probchg_crop',]) +
  tm_fill('probchg', palette = 'RdYlGn', style = 'quantile',
          n = 5, title = 'Change in Probability\nof Converting to Crop Use', legend.hist = F) +
  tm_layout(scale = 0.8, legend.position = c('right','bottom'),
            legend.outside = T) +
  tm_shape(mapd[mapd$type_landuse == 'full_impact_startfr_probchg_crop',]) +
  tm_borders(lwd = .5, alpha = .5)

map6 <- tm_shape(mapd[mapd$type_landuse == 'full_impact_startfr_probchg_forest',]) +
  tm_fill('probchg', palette = 'RdYlGn', style = 'quantile',
          n = 5, title = 'Change in Probability\nof Remaining in Forest Use', legend.hist = F) +
  tm_layout(scale = 0.8, legend.position = c('right','bottom'),
            legend.outside = T) +
  tm_shape(mapd[mapd$type_landuse == 'full_impact_startfr_probchg_forest',]) +
  tm_borders(lwd = .5, alpha = .5)

map7 <- tm_shape(mapd[mapd$type_landuse == 'full_impact_startfr_probchg_urban',]) +
  tm_fill('probchg', palette = 'RdYlGn', style = 'quantile',
          n = 5, title = 'Change in Probability\nof Converting to Urban Use', legend.hist = F) +
  tm_layout(scale = 0.8, legend.position = c('right','bottom'),
            legend.outside = T) +
  tm_shape(mapd[mapd$type_landuse == 'full_impact_startfr_probchg_urban',]) +
  tm_borders(lwd = .5, alpha = .5)

map8 <- tm_shape(mapd[mapd$type_landuse == 'full_impact_startfr_probchg_pasture',]) +
  tm_fill('probchg', palette = 'RdYlGn', style = 'quantile',
          n = 5, title = 'Change in Probability\nof Converting to Pasture Use', legend.hist = F) +
  tm_layout(scale = 0.8, legend.position = c('right','bottom'),
            legend.outside = T) +
  tm_shape(mapd[mapd$type_landuse == 'full_impact_startfr_probchg_pasture',]) +
  tm_borders(lwd = .5, alpha = .5)

pdf('cc_impact_map_crstart.pdf')
tmap_arrange(map1, map2, map3, map4, nrow = 4)
dev.off()

pdf('cc_impact_map_frstart.pdf')
tmap_arrange(map5, map6, map7, map8, nrow = 4)
dev.off()
