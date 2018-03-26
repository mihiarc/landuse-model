# calculate marginal effects for land use model

library(tidyverse)
library(tmap)
library(sf)

#load functions-----------------------------------------------------------------

calc_utility <- function(model, df) {
  
  df$crop_util <- model$coefficients[7] * df$nr.cr +
    model$coefficients[8] * df$nrchange.cr +
    model$coefficients[9] * df$nr.cr *
    df$nrchange.cr
  
  df$pasture_util <- model$coefficients[3] + model$coefficients[4] *
    df$lcc + model$coefficients[13] *
    df$nr.ps + model$coefficients[14] *
    df$nrchange.ps + model$coefficients[15] *
    df$nr.ps * df$nrchange.ps
  
  df$forest_util <- model$coefficients[1] + model$coefficients[2] *
    df$lcc + model$coefficients[10] *
    df$nr.fr + model$coefficients[11] *
    df$nrchange.fr + model$coefficients[12] *
    df$nr.fr * df$nrchange.fr
  
  df$urban_util <- model$coefficients[5] + model$coefficients[6] *
    df$lcc + model$coefficients[16] *
    df$nr.ur + model$coefficients[17] *
    df$nrchange.ur + model$coefficients[18] *
    df$nr.ur * df$nrchange.ur
  
  df
}

calc_prob <- function(model, df1, df2) {
  
  df1 <- calc_utility(model, df1)
  df2 <- calc_utility(model, df2)
  
  df1 <- df1 %>%
    mutate(pr_crop = exp(crop_util) / (exp(crop_util) + exp(pasture_util) + 
                                         exp(forest_util) + exp(urban_util)),
           pr_pasture = exp(pasture_util) / (exp(crop_util) + exp(pasture_util) + 
                                     exp(forest_util) + exp(urban_util)),
           pr_forest = exp(forest_util) / (exp(crop_util) + exp(pasture_util) + 
                                             exp(forest_util) + exp(urban_util)),
           pr_urban = exp(urban_util) / (exp(crop_util) + exp(pasture_util) + 
                                           exp(forest_util) + exp(urban_util)))
  df2 <- df2 %>%
    mutate(pr_crop = exp(crop_util) / (exp(crop_util) + exp(pasture_util) + 
                                         exp(forest_util) + exp(urban_util)),
           pr_pasture = exp(pasture_util) / (exp(crop_util) + exp(pasture_util) + 
                                          exp(forest_util) + exp(urban_util)),
           pr_forest = exp(forest_util) / (exp(crop_util) + exp(pasture_util) + 
                                             exp(forest_util) + exp(urban_util)),
           pr_urban = exp(urban_util) / (exp(crop_util) + exp(pasture_util) + 
                                           exp(forest_util) + exp(urban_util)))
  
  df <- df1 %>%
    select(riad_id, starts_with('pr')) %>%
    inner_join(select(df2, riad_id, xfact, starts_with('pr')), by = 'riad_id') %>%
    mutate(mfx_crop = pr_crop.y - pr_crop.x, 
           mfx_pasture = pr_pasture.y - pr_pasture.x,
           mfx_forest = pr_forest.y - pr_forest.x,
           mfx_urban = pr_urban.y - pr_urban.x) %>%
    select(riad_id, xfact, starts_with('mfx'))
  
  df
}

pred_data <- function(choice_data, nr_data, georef) {
  df <- inner_join(choice_data, nr_data,  by = c('fips','year')) %>%
    inner_join(georef[c('fips','subregion','region','mer100')], by = 'fips') %>%
    filter(year == 2012 & mer100 == 'east') %>%
    select(-ends_with('ag')) %>%
    filter(!is.na(nr.fr), !is.na(nrchange.fr))
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

# load data--------------------------------------------------------------------

georef <- tbl_df(read.csv("forest_georef.csv"))
georef <- rename(georef, fips = county_fips)
georef$fips <- as.integer(georef$fips)

nr_data <- tbl_df(readRDS('nr_clean_5year_normals.rds'))
nr_data$year <- nr_data$year + 2 # calculate lagged nr
nr_data_crnr_mfx <- tbl_df(readRDS('nr_clean_5year_normals_crnr_mfx.rds'))
nr_data_crnr_mfx$year <- nr_data_crnr_mfx$year + 2 # calculate lagged nr
nr_data_psnr_mfx <- tbl_df(readRDS('nr_clean_5year_normals_psnr_mfx.rds'))
nr_data_psnr_mfx$year <- nr_data_psnr_mfx$year + 2 # calculate lagged nr
nr_data_frnr_mfx <- tbl_df(readRDS('nr_clean_5year_normals_frnr_mfx.rds'))
nr_data_frnr_mfx$year <- nr_data_frnr_mfx$year + 2 # calculate lagged nr
nr_data_urnr_mfx <- tbl_df(readRDS('nr_clean_5year_normals_urnr_mfx.rds'))
nr_data_urnr_mfx$year <- nr_data_frnr_mfx$year + 2 # calculate lagged nr

start_crop <- tbl_df(readRDS('start_crop.rds'))
start_crop$lcc <- as.numeric(start_crop$lcc)
start_pasture <- tbl_df(readRDS('start_pasture.rds'))
start_pasture$lcc <- as.numeric(start_pasture$lcc)
start_forest <- tbl_df(readRDS('start_forest.rds'))
start_forest$lcc <- as.numeric(start_forest$lcc)

# merge and format data---------------------------------------------------------
# function to format prediction data



crstart <- list(no_chg = pred_data(start_crop, nr_data, georef),
                crnr_chg = pred_data(start_crop, nr_data_crnr_mfx,
                                     georef),
                psnr_chg = pred_data(start_crop, nr_data_psnr_mfx,
                                     georef),
                frnr_chg = pred_data(start_crop, nr_data_frnr_mfx,
                                     georef),
                urnr_chg = pred_data(start_crop, nr_data_urnr_mfx,
                                     georef))

psstart <- list(no_chg = pred_data(start_pasture, nr_data, georef),
                crnr_chg = pred_data(start_pasture, nr_data_crnr_mfx,
                                     georef),
                psnr_chg = pred_data(start_pasture, nr_data_psnr_mfx,
                                     georef),
                frnr_chg = pred_data(start_pasture, nr_data_frnr_mfx,
                                     georef),
                urnr_chg = pred_data(start_pasture, nr_data_urnr_mfx,
                                     georef))

frstart <- list(no_chg = pred_data(start_forest, nr_data, georef),
                crnr_chg = pred_data(start_forest, nr_data_crnr_mfx,
                                     georef),
                psnr_chg = pred_data(start_forest, nr_data_psnr_mfx,
                                     georef),
                frnr_chg = pred_data(start_forest, nr_data_frnr_mfx,
                                     georef),
                urnr_chg = pred_data(start_forest, nr_data_urnr_mfx,
                                     georef))

# load and predict models-------------------------------------------------------

# landuse_models <- readRDS('crop_forest_lvl_chg.rds')
model_crstart <- landuse_models[[2]]
model_psstart <- landuse_models[[6]]
model_frstart <- landuse_models[[10]]

mfx_crstart <- list(crnr_mfx = calc_prob(model_crstart, crstart[[1]], crstart[[2]]),
                    psnr_mfx = calc_prob(model_crstart, crstart[[1]], crstart[[3]]),
                    frnr_mfx = calc_prob(model_crstart, crstart[[1]], crstart[[4]]),
                    urnr_mfx = calc_prob(model_crstart, crstart[[1]], crstart[[5]]))

mfx_psstart <- list(crnr_mfx = calc_prob(model_psstart, psstart[[1]], psstart[[2]]),
                    psnr_mfx = calc_prob(model_psstart, psstart[[1]], psstart[[3]]),
                    frnr_mfx = calc_prob(model_psstart, psstart[[1]], psstart[[4]]),
                    urnr_mfx = calc_prob(model_psstart, psstart[[1]], psstart[[5]]))

mfx_frstart <- list(crnr_mfx = calc_prob(model_frstart, frstart[[1]], frstart[[2]]),
                    psnr_mfx = calc_prob(model_frstart, frstart[[1]], frstart[[3]]),
                    frnr_mfx = calc_prob(model_frstart, frstart[[1]], frstart[[4]]),
                    urnr_mfx = calc_prob(model_frstart, frstart[[1]], frstart[[5]]))

# load and merge georeference
nri_georef <- readRDS('nri_georef.rds')
nri_georef$fips <- str_pad(as.character(nri_georef$fips),
                           width = 5, side = 'left', pad = '0')

mfx_crstart <- map(mfx_crstart, ~ inner_join(., nri_georef, by = 'riad_id'))
mfx_psstart <- map(mfx_psstart, ~ inner_join(., nri_georef, by = 'riad_id'))
mfx_frstart <- map(mfx_frstart, ~ inner_join(., nri_georef, by = 'riad_id'))

mfx <- list(crstart = mfx_crstart,
                   psstart = mfx_psstart,
                   frstart = mfx_frstart)

saveRDS(mfx, 'crop_forest_mfx.rds')

# function for aggregating to county level mfx-------------------------------------------------



mfx_crstart_county <- map(mfx_crstart, ~ aggr_to_county(.))
mfx_psstart_county <- map(mfx_psstart, ~ aggr_to_county(.))
mfx_frstart_county <- map(mfx_frstart, ~ aggr_to_county(.))

mfx_county <- list(crstart = mfx_crstart_county,
            psstart = mfx_psstart_county,
            frstart = mfx_frstart_county)

saveRDS(mfx_county, 'crop_forest_mfx_county.rds')

map(mfx_crstart, ~ summary(.))
map(mfx_crstart_county, ~ summary(.))

map(mfx_psstart, ~ summary(.))
map(mfx_psstart_county, ~ summary(.))

map(mfx_frstart, ~ summary(.))
map(mfx_frstart_county, ~ summary(.))

# maps--------------------------------------------------------------------------

counties <- st_read('D:/GroupWork/Climate Data/Spatial Files/Census Boundaries/conus_county.shp')

# map function------------------------------------------------------------------

map_ame <- function(counties, df_mfx, chg_var, legend_title = 'AME') {
  df <- inner_join(counties, df_mfx, by = c('GEOID' = 'fips'))
  df$GEOID <- as.factor(df$GEOID)
  
  tm_shape(df) +
    tm_fill(chg_var, palette = "RdYlGn", style = 'quantile', n = 9,
            title = legend_title, legend.hist = F) +
    tm_shape(df) +
    tm_layout(legend.outside = T) +
    tm_borders(lwd = .5, alpha = .5) +
    tm_layout(legend.outside = T)
}

crnr_crmfx_crstart <- map_ame(counties, mfx_crstart_county[[1]], 'mfx_crop')
psnr_psmfx_crstart <- map_ame(counties, mfx_crstart_county[[2]], 'mfx_pasture')
frnr_frmfx_crstart <- map_ame(counties, mfx_crstart_county[[3]], 'mfx_forest')
urnr_urmfx_crstart <- map_ame(counties, mfx_crstart_county[[4]], 'mfx_urban')

crnr_crmfx_psstart <- map_ame(counties, mfx_psstart_county[[1]], 'mfx_crop')
psnr_psmfx_psstart <- map_ame(counties, mfx_psstart_county[[2]], 'mfx_pasture')
frnr_frmfx_psstart <- map_ame(counties, mfx_psstart_county[[3]], 'mfx_forest')
urnr_urmfx_psstart <- map_ame(counties, mfx_psstart_county[[4]], 'mfx_urban')

crnr_crmfx_frstart <- map_ame(counties, mfx_frstart_county[[1]], 'mfx_crop')
psnr_psmfx_frstart <- map_ame(counties, mfx_frstart_county[[2]], 'mfx_pasture')
frnr_frmfx_frstart <- map_ame(counties, mfx_frstart_county[[3]], 'mfx_forest')
urnr_urmfx_frstart <- map_ame(counties, mfx_frstart_county[[4]], 'mfx_urban')

#-------------------------------------------------------------------------------


# ag start frnr mfx county
ggplot(frnr_mfx_county, aes(mfx_forest)) +
  geom_histogram(breaks = seq(min(frnr_mfx_county$mfx_forest),
                              max(frnr_mfx_county$mfx_forest), by = .00001),
                 col = 'black',
                 fill = 'green',
                 alpha = .2) +
  xlim(-0.00005, 3e-04) +
  labs(x = 'County Average Marginal Effect', title = 'Change in Forest NR for land starting in Ag')

#ag start agnr mfx county
ggplot(agnr_mfx_county, aes(mfx_ag)) +
  geom_histogram(breaks = seq(min(agnr_mfx_county$mfx_ag),
                              max(agnr_mfx_county$mfx_ag),
                              by = abs(mean(agnr_mfx_county$mfx_ag))),
                 col = 'black',
                 fill = 'green',
                 alpha = .2) +
  xlim(-.001, max(agnr_mfx_county$mfx_ag)) +
  labs(x = 'County Average Marginal Effect', title = 'Change in Ag NR for land starting in Ag')

#ag start agnr mfx county
ggplot(urnr_mfx_county, aes(mfx_urban)) +
  geom_histogram(breaks = seq(min(urnr_mfx_county$mfx_urban),
                              max(urnr_mfx_county$mfx_urban),
                              by = .001),
                 col = 'black',
                 fill = 'green',
                 alpha = .2) +
  xlim(-.01,.013) +
  labs(x = 'County Average Marginal Effect', title = 'Change in Urban NR for land starting in Ag')
