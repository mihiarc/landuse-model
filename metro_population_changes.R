# consider metro areas
library(tidyverse)
library(sf)
library(tmap)
library(ggplot2)

counties <- st_read('shapefiles/conus_county.shp')
counties$fips <- as.character(counties$GEOID)

d1 <- tbl_df(read.csv('pop_change_data.csv')) %>%
  rename('fips' = 'STCOU') %>%
  filter(!is.na(fips)) %>%
  mutate(fips = str_pad(as.character(fips), width = 5,
                        pad = '0', side = 'left'))

saveRDS(unique(d1$fips), 'metro_fips.rds')


# create map of population percent change in metro counties---------------------

df <- inner_join(counties, d1, by = 'fips')

tm_shape(df) +
  tm_fill('PERCENT', palette = "RdYlGn", style = 'quantile',
          n = 9, legend.hist = F) +
  tm_shape(counties) +
  tm_layout(legend.outside = T) +
  tm_borders(lwd = .5, alpha = .5) +
  tm_layout(legend.outside = T)



