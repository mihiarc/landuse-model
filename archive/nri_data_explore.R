# create maps of land use shares from nri data

library(tidyverse)
library(sf)
library(tmap)

lu <- tbl_df(readRDS('landuse_shares.rds'))
lu$fips <- factor(str_pad(lu$fips, 5, side = 'left', pad = '0'))
counties <- st_read('shapefiles/conus_county.shp')

d1 <- counties %>%
	left_join(lu, by = c('GEOID' = 'fips'))


######################################################################
######################################################################

frmap <- tm_shape(filter(d1, mer100 == 'east')) +
	tm_fill('fr_acres_change', palette = "RdYlGn", style = 'quantile', n = 5, title = 'Change in Forest Acres', legend.hist = T) +
	tm_layout(scale = 0.8, legend.position = c('right','bottom'), legend.outside = T) +
	tm_shape(filter(d1, mer100 == 'east')) +
	tm_borders(lwd = .5, alpha = .5)

agmap <- tm_shape(filter(d1, mer100 == 'east')) +
	tm_fill('ag_acres_change', palette = "RdYlGn", style = 'quantile', n = 7, title = 'Change in Ag Acres', legend.hist = T) +
	tm_layout(scale = 0.8, legend.position = c('left','bottom'), legend.outside = T) +
	tm_shape(filter(d1, mer100 == 'east')) +
	tm_borders(lwd = .5, alpha = .5)

urmap <- tm_shape(filter(d1, mer100 == 'east')) +
	tm_fill('ur_acres_change', palette = "RdYlGn", style = 'quantile', n = 7, title = 'Change in Urban Acres', legend.hist = T) +
	tm_layout(scale = 0.8, legend.position = c('left','bottom'), legend.outside = T) +
	tm_shape(filter(d1, mer100 == 'east')) +
	tm_borders(lwd = .5, alpha = .5)

tmap_arrange(frmap, agmap, urmap, asp = NA)

######################################################################

