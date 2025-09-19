# discrete choice logit

library(mlogit)
library(mnlogit)
library(tidyverse)

# load data--------------------------------------------------------------------

georef <- tbl_df(read.csv("forest_georef.csv"))
georef <- rename(georef, fips = county_fips)
georef$fips <- as.integer(georef$fips)

nr_data <- tbl_df(readRDS('nr_clean_5year_normals.rds'))
# calc lagged net returns------------------------------------------------------
nr_data$year <- nr_data$year + 2

start_crop <- tbl_df(readRDS('start_crop.rds'))
start_pasture <- tbl_df(readRDS('start_pasture.rds'))
start_forest <- tbl_df(readRDS('start_forest.rds'))

# merge data--------------------------------------------------------------------

#crop start
ec <- inner_join(start_crop, nr_data, by = c('fips','year')) %>%
  inner_join(georef[c('fips','subregion','region')], by = 'fips') %>%
  select(-ends_with('ag'))
ec$lcc <- as.integer(as.character(ec$lcc))
ec <- filter(ec, year %in% c(2010:2012))
# ec$lcc_group <- 0
# ec$lcc_group[ec$lcc %in% c(1:2)] <- 1
# ec$lcc_group[ec$lcc %in% c(3:4)] <- 2
# ec$lcc_group[ec$lcc %in% c(5:6)] <- 3
# ec$lcc_group[ec$lcc %in% c(7:8)] <- 4

ec_east <- filter(ec, region %in% c('NO','SO'))
ec_south <- filter(ec, region == 'SO')
ec_north <- filter(ec, region == 'NO')

ec <- cbind(ec, weight = (ec$xfact / sum(ec[,"xfact"])) * nrow(ec))
ec_east <- cbind(ec_east, weight = (ec_east$xfact / sum(ec_east[,"xfact"])) * nrow(ec_east))
ec_south <- cbind(ec_south, weight = (ec_south$xfact / sum(ec_south[,"xfact"])) * nrow(ec_south))
ec_north <- cbind(ec_north, weight = (ec_north$xfact / sum(ec_north[,"xfact"])) * nrow(ec_north))

#pasture start------------------------------------------------------------------
ep <- inner_join(start_pasture, nr_data, by = c('fips','year')) %>%
  inner_join(georef[c('fips','subregion','region')], by = 'fips') %>%
  select(-ends_with('ag'))
ep$lcc <- as.integer(as.character(ep$lcc))
ep <- filter(ep, year %in% c(2010:2012))
# ep$lcc_group <- 0
# ep$lcc_group[ep$lcc %in% c(1:2)] <- 1
# ep$lcc_group[ep$lcc %in% c(3:4)] <- 2
# ep$lcc_group[ep$lcc %in% c(5:6)] <- 3
# ep$lcc_group[ep$lcc %in% c(7:8)] <- 4

ep_east <- filter(ep, region %in% c('NO','SO'))
ep_south <- filter(ep, region == 'SO')
ep_north <- filter(ep, region == 'NO')

ep <- cbind(ep, weight = (ep$xfact / sum(ep[,"xfact"])) * nrow(ep))
ep_east <- cbind(ep_east, weight = (ep_east$xfact / sum(ep_east[,"xfact"])) * nrow(ep_east))
ep_south <- cbind(ep_south, weight = (ep_south$xfact / sum(ep_south[,"xfact"])) * nrow(ep_south))
ep_north <- cbind(ep_north, weight = (ep_north$xfact / sum(ep_north[,"xfact"])) * nrow(ep_north))

# forest start-----------------------------------------------------------------

ef <- inner_join(start_forest, nr_data, by = c('fips','year')) %>%
  inner_join(georef[c('fips','subregion','region')], by = 'fips') %>%
  select(-ends_with('ag'))
ef$lcc <- as.integer(as.character(ef$lcc))
ef <- filter(ef, year %in% c(2010:2012))
# ef$lcc_group <- 0
# ef$lcc_group[ef$lcc %in% c(1:2)] <- 1
# ef$lcc_group[ef$lcc %in% c(3:4)] <- 2
# ef$lcc_group[ef$lcc %in% c(5:6)] <- 3
# ef$lcc_group[ef$lcc %in% c(7:8)] <- 4


ef_east <- filter(ef, region %in% c('NO','SO'))
ef_south <- filter(ef, region == 'SO')
ef_north <- filter(ef, region == 'NO')

ef <- cbind(ef, weight = (ef$xfact / sum(ef[,"xfact"])) * nrow(ef))
ef_east <- cbind(ef_east, weight = (ef_east$xfact / sum(ef_east[,"xfact"])) * nrow(ef_east))
ef_south <- cbind(ef_south, weight = (ef_south$xfact / sum(ef_south[,"xfact"])) * nrow(ef_south))
ef_north <- cbind(ef_north, weight = (ef_north$xfact / sum(ef_north[,"xfact"])) * nrow(ef_north))

# create list of all data frames------------------------------------------------



est_data <- list(cropstart_united_states = ec,
                 cropstart_east = ec_east,
                 cropstart_south = ec_south,
                 cropstart_north = ec_north,
                 pasturestart_united_states = ep,
                 pasturestart_east = ep_east,
                 pasturestart_south = ep_south,
                 pasturestart_north = ep_north,
                 foreststart_united_states = ef,
                 foreststart_east = ef_east,
                 foreststart_south = ef_south,
                 foreststart_north = ef_north)

# saveRDS(est_data, 'estimation_data_crop_forest_logit.rds')


# format data for mlogit-------------------------------------------------------


d <- map(est_data, ~ mlogit.data(as.data.frame(.), shape = 'wide',
                                    choice = 'choice', varying = c(7:22)))

d <- map(d, ~ list(., group_by(., riad_id, year) %>% summarize(weight = unique(weight)) %>% ungroup()))

# clean up workspace------------------------------------------------------------

rm(ec, ec_east, ec_north, ec_south, ef, ef_east, ef_north, ef_south,
   ep, ep_east, ep_north, ep_south, est_data, georef, nr_data, start_crop,
   start_forest, start_pasture)

# fit models--------------------------------------------------------------------

fit_nr_levels <- map(d, ~ mnlogit(choice ~ 1 | lcc | nr, .[[1]],
                                  weights = .[[2]][[3]]))
fit_nr_change <- map(d, ~ mnlogit(choice ~ 1 | lcc | nrchange, .[[1]],
                                  weights = .[[2]][[3]]))
fit_nr_lvl_chg <- map(d, ~ mnlogit(choice ~ 1 | lcc | nr + nrchange +
                                     I(nr * nrchange), .[[1]],
                                   weights = .[[2]][[3]]))
fit_nr_means <- map(d, ~ mnlogit(choice ~ 1 | lcc | nr + nrmean, .[[1]],
                                 weights = .[[2]][[3]]))
fit_nrchange_means <- map(d, ~ mnlogit(choice ~ 1 | lcc | nrchange +
                                         nrchangemean, .[[1]],
                                       weights = .[[2]][[3]]))
fit_nr_all <- map(d, ~ mnlogit(choice ~ 1 | lcc | nr + nrchange +
                                 I(nr * nrchange) + nrmean + nrchangemean,
                               .[[1]], weights = .[[2]][[3]]))

# restrict estimation to 2010, 2012 transition periods only

fit_nr_levels_1012 <- map(d, ~ mnlogit(choice ~ 1 | lcc_group | nr, .[[1]],
                                  weights = .[[2]][[3]]))
map(fit_nr_levels_1012, ~ summary(.))

# save model estimates---------------------------------------------------------

saveRDS(fit_nr_levels, 'crop_forest_levels.rds')
saveRDS(fit_nr_change, 'crop_forest_change.rds')
saveRDS(fit_nr_lvl_chg, 'crop_forest_lvl_chg.rds')
saveRDS(fit_nr_means, 'crop_forest_levels_means.rds')
saveRDS(fit_nrchange_means, 'crop_forest_change_means.rds')
saveRDS(fit_nr_all, 'crop_forest_all.rds')
saveRDS(fit_nr_levels_1012, 'crop_forest_levels_2010_2012.rds')

# reload models-----------------------------------------------------------------

fit_nr_levels <- readRDS('crop_forest_levels.rds')
fit_nr_change <- readRDS('crop_forest_change.rds')
fit_nr_lvl_chg <- readRDS('crop_forest_lvl_chg.rds')
fit_nr_means <- readRDS('crop_forest_levels_means.rds')
fit_nrchange_means <- readRDS('crop_forest_change_means.rds')
fit_nr_all <- readRDS('crop_forest_all.rds')

# calculate marginal effects for models that include an interaction-------------

mfx_crnr <- map(fit_nr_lvl_chg, ~ logit_mfx(., alt_name = 'cr', chg_var = 'nr'))
mfx_psnr <- map(fit_nr_lvl_chg, ~ logit_mfx(., alt_name = 'ps', chg_var = 'nr'))
mfx_frnr <- map(fit_nr_lvl_chg, ~ logit_mfx(., alt_name = 'fr', chg_var = 'nr'))
mfx_urnr <- map(fit_nr_lvl_chg, ~ logit_mfx(., alt_name = 'ur', chg_var = 'nr'))

mfx_crnr2 <- map(fit_nr_all, ~ logit_mfx(., alt_name = 'cr', chg_var = 'nr'))
mfx_psnr2 <- map(fit_nr_all, ~ logit_mfx(., alt_name = 'ps', chg_var = 'nr'))
mfx_frnr2 <- map(fit_nr_all, ~ logit_mfx(., alt_name = 'fr', chg_var = 'nr'))
mfx_urnr2 <- map(fit_nr_all, ~ logit_mfx(., alt_name = 'ur', chg_var = 'nr'))

# average marginal effects of change in crop nr
mfx_crop_nr_lvl_chg <- map(fit_nr_lvl_chg, ~ krob_mfx(., alt_name = 'cr', chg_var = 'nr'))


# average marginal effects of change in forest nr
mfx_forest_nr_lvl_chg <- map(fit_nr_lvl_chg, ~ krob_mfx(., alt_name = 'fr', chg_var = 'nr'))

#pasture
mfx_pasture_nr_lvl_chg <- map(fit_nr_lvl_chg, ~ krob_mfx(., alt_name = 'ps', chg_var = 'nr'))

#urban
mfx_urban_nr_lvl_chg <- map(fit_nr_lvl_chg, ~ krob_mfx(., alt_name = 'ur', chg_var = 'nr'))


crop_forest_mfx <- list(crop_nr_lvl_chg = mfx_crop_nr_lvl_chg,
                        forest_nr_lvl_chg = mfx_forest_nr_lvl_chg,
                        pasture_nr_lvl_chg = mfx_pasture_nr_lvl_chg,
                        urban_nr_lvl_chg = mfx_urban_nr_lvl_chg)

saveRDS(crop_forest_mfx, 'crop_forest_mfx_lvl_chg.rds')